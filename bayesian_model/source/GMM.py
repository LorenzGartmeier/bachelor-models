import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class GMM:
    def __init__(self, num_components, selfdescription_length, num_samples=30):
        self.num_components = num_components
        self.dim = selfdescription_length
        self.num_samples = num_samples  # MC samples for predictions
        
        # Define parameter shapes
        locs_shape = [num_components, selfdescription_length]
        scale_shape = [num_components, selfdescription_length]
        mix_shape = [num_components]

        # Prior distributions (weakly informative)
        self.prior = tfd.JointDistributionNamed({
            'locs': tfd.Independent(
                tfd.Normal(tf.zeros(locs_shape), 
                reinterpreted_batch_ndims=2)),
            'scale_diag_raw': tfd.Independent(
                tfd.Normal(tf.zeros(scale_shape)), 
                reinterpreted_batch_ndims=2),
            'mix_probs_raw': tfd.Independent(
                tfd.Normal(tf.zeros(mix_shape)),
                reinterpreted_batch_ndims=1)
        })

        # Build surrogate posterior (mean-field approximation)
        self.surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=self.prior.event_shape,
            bijector={
                'locs': tfb.Identity(),
                'scale_diag_raw': tfb.Identity(),
                'mix_probs_raw': tfb.Identity()
            },
            initial_loc={
                'locs': tf.random.normal(locs_shape, stddev=0.1),
                'scale_diag_raw': tf.random.normal(scale_shape, stddev=0.1),
                'mix_probs_raw': tf.random.normal(mix_shape, stddev=0.1)
            }
        )

    def fit(self, dataset, epochs, learning_rate=0.05):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        def generative_model():
            # returns a dictionary with keys: 'locs', 'scale_diag_raw', 'mix_probs_raw'

            params = yield self.prior
            scale_diag = tf.nn.softplus(params['scale_diag_raw'])
            mix_probs = tf.nn.softmax(params['mix_probs_raw'])
            gmm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mix_probs),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=params['locs'], 
                    scale_diag=scale_diag
                )
            )
            yield tfd.Independent(gmm, reinterpreted_batch_ndims=1)
        
        for epoch in range(epochs):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    # Compute ELBO loss
                    loss = -tf.reduce_mean(
                        tfp.vi.monte_carlo_variational_loss(
                            target_log_prob_fn=lambda *args: generative_model().log_prob(batch, *args),
                            surrogate_posterior=self.surrogate_posterior,
                            sample_size=self.num_samples,  # ELBO samples
                            seed=epoch
                        )
                    )
                
                # Apply gradients
                grads = tape.gradient(loss, self.surrogate_posterior.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.surrogate_posterior.trainable_variables))

    def get_prob(self, x):
        """Compute predictive probability via MC integration"""
        posterior_samples = self.surrogate_posterior.sample(self.num_samples)
        
        # Compute probabilities for all samples
        probs = []
        for i in range(self.num_samples):
            scale_diag = tf.nn.softplus(posterior_samples['scale_diag_raw'][i])
            mix_probs = tf.nn.softmax(posterior_samples['mix_probs_raw'][i])
            gmm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mix_probs),
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=posterior_samples['locs'][i],
                    scale_diag=scale_diag
                )
            )
            probs.append(gmm.prob(x))
        
        # Average over samples
        return tf.reduce_mean(tf.stack(probs, axis=0), tf.math.reduce_std(tf.stack(probs, axis=0)))
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


class GMM:
    def __init__(self, num_components, selfdescription_lengh):
        self.num_components = num_components
        self.locs = tf.Variable(tf.random.normal([num_components, selfdescription_lengh]))
        self.scale_diag_raw = tf.Variable(tf.random.normal([num_components, selfdescription_lengh]))
        self.mix_probs_raw = tf.Variable(tf.random.normal([num_components]))

    def fit(self, dataset, epochs, learning_rate=0.1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for epoch in range(epochs):
            for batch in dataset:

                # Perform one step of optimization
                with tf.GradientTape() as tape:
                    scale_diag = tf.nn.softplus(self.scale_diag_raw)
                    mix_probs = tf.nn.softmax(self.mix_probs_raw)
                    gmm = tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=mix_probs),
                        components_distribution=tfd.MultivariateNormalDiag(loc=self.locs, scale_diag=scale_diag),
                    )
                    loss = -tf.reduce_mean(gmm.log_prob(batch))
                grads = tape.gradient(loss, [self.locs, self.scale_diag_raw, self.mix_probs_raw])
                optimizer.apply_gradients(zip(grads, [self.locs, self.scale_diag_raw, self.mix_probs_raw]))


    def get_prob(self, batch):
        scale_diag = tf.nn.softplus(self.scale_diag_raw)
        mix_probs = tf.nn.softmax(self.mix_probs_raw)
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix_probs),
            components_distribution=tfd.MultivariateNormalDiag(loc=self.locs, scale_diag=scale_diag),
        )
        return gmm.prob(batch)
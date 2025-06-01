import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


class GMM:
    def __init__(self, num_components):
        self.num_components = num_components
        self.locs = tf.Variable(tf.random.normal([num_components, 2]))
        self.scale_diag = tf.nn.softplus(tf.Variable(tf.random.normal([num_components, 2])))
        self.mix_probs = tf.nn.softmax(tf.Variable(tf.random.normal([num_components])))

    def fit(self, dataset, epochs, learning_rate=0.1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for epoch in range(epochs):
            for batch in dataset:

                # Perform one step of optimization
                with tf.GradientTape() as tape:
                    gmm = tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=self.mix_probs),
                        components_distribution=tfd.MultivariateNormalDiag(loc=self.locs, scale_diag=self.scale_diag),
                    )
                    loss = -tf.reduce_mean(gmm.log_prob(batch))
                grads = tape.gradient(loss, [self.locs, self.scale_diag, self.mix_probs])
                optimizer.apply_gradients(zip(grads, [self.locs, self.scale_diag, self.mix_probs]))


    def get_prob(self, btach):
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.mix_probs),
            components_distribution=tfd.MultivariateNormalDiag(loc=self.locs, scale_diag=self.scale_diag),
        )
        return gmm.prob(btach)
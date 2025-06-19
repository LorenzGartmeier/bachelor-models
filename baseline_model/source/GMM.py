import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

class GMM(tf.Module):
    def __init__(self, num_components: int, vec_len: int):
        super().__init__()
        self.num_components = num_components
        self.vec_len        = vec_len          # <— store for save/load

        # parameters
        self.locs      = tf.Variable(tf.random.normal([num_components, vec_len]))
        self.scale_raw = tf.Variable(tf.random.normal([num_components, vec_len]))
        self.mix_raw   = tf.Variable(tf.random.normal([num_components]))

    # helper that always applies σ-floor
    def _dist(self):
        scale = tf.nn.softplus(self.scale_raw) + 1e-4
        mix   = tf.nn.softmax(self.mix_raw)
        return tfd.MixtureSameFamily(
            tfd.Categorical(probs=mix),
            tfd.MultivariateNormalDiag(loc=self.locs, scale_diag=scale))

    def fit(self, ds, epochs, lr=1e-2, clip=10.0):
        opt = tf.keras.optimizers.Adam(lr)
        history = {
            'loss': [],
        }

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in ds:
                with tf.GradientTape() as tape:
                    loss = -tf.reduce_mean(self._dist().log_prob(batch))
                g = tape.gradient(loss, self.trainable_variables)
                opt.apply_gradients(zip(g, self.trainable_variables))
                epoch_loss += loss
            epoch_loss /= tf.data.experimental.cardinality(ds).numpy()
            history['loss'].append(epoch_loss.numpy())
        return history

    def log_prob(self, x):  return self._dist().log_prob(x)
    def prob    (self, x):  return tf.exp(self.log_prob(x))

    def save(self, path):
        np.savez(path,
                 locs=self.locs.numpy(),
                 scale_raw=self.scale_raw.numpy(),
                 mix_raw=self.mix_raw.numpy(),
                 num_components=self.num_components,
                 vec_len=self.vec_len)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(int(d['num_components']), int(d['vec_len']))
        obj.locs     .assign(d['locs'])
        obj.scale_raw.assign(d['scale_raw'])
        obj.mix_raw  .assign(d['mix_raw'])
        return obj

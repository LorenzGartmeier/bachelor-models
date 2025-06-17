import numpy as np
import tensorflow as tf
from GMM import GMM  # Assuming GMM is in the same directory or properly imported    

class Attributor:
    def __init__(self, n_components):
        self.n_components          = n_components
        self.gmm_dict: dict[int, GMM] = {}

    # returns loss history
    def add_gmm(self, label: int, dataset, epochs=10, lr=1e-2):

        for batch in dataset.take(1):
            selfdescription_length = batch.shape[-1]
        gmm = self.gmm_dict.setdefault(
                  label, GMM(self.n_components, selfdescription_length))
        return gmm.fit(dataset, epochs=epochs, lr=lr)

    def predict(self, batch, tau: float):
        x = tf.convert_to_tensor(batch, tf.float32)

        best_logprob  = tf.fill([x.shape[0]], -np.inf)
        best_label = tf.fill([x.shape[0]], -1)

        for label, gmm in self.gmm_dict.items():
            logprob = gmm.log_prob(x)             # (B,)
            mask = logprob > best_logprob
            best_logprob  = tf.where(mask, logprob, best_logprob)
            best_label = tf.where(mask, label, best_label)

        best_label = tf.where(best_logprob >= tau, best_label, -1)
        return best_label.numpy()

    def save(self, base_path: str):
        meta = dict(n_components=self.n_components,
                    vec_len=self.selfdescription_length,
                    labels=[])
        for label, gmm in self.gmm_dict.items():
            gmm.save(f"{base_path}_{label}.npz")
            meta['labels'].append(label)
        np.savez(f"{base_path}_meta.npz", **meta)

    @classmethod
    def load(cls, base_path: str):
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        atr  = cls(int(meta['n_components']), int(meta['vec_len']))
        for label in meta['labels']:
            atr.gmm_dict[int(label)] = GMM.load(f"{base_path}_{label}.npz")
        return atr






import os
import sys
project_root = os.environ.get('BACHELOR_MODELS_ROOT', '.')
sys.path.append(project_root)

import numpy as np
import tensorflow as tf
from tensorflow import math
from baseline_model.source.GMM import GMM

class Attributor:
    def __init__(self, num_components):
        self.num_components = num_components
        self.gmm_dict: dict[int, GMM] = {}

    # returns loss history
    def add_gmm(self, label: int, dataset, epochs=10, lr=1e-2):

        for batch in dataset.take(1):
            selfdescription_length = batch.shape[-1]
        gmm = self.gmm_dict.setdefault(
                  label, GMM(self.num_components, selfdescription_length))
        return gmm.fit(dataset, epochs=epochs, lr=lr)

    def save(self, base_path: str):
        meta = dict(n_components=self.num_components,
                    labels=[])
        for label, gmm in self.gmm_dict.items():
            gmm.save(f"{base_path}_{label}.npz")
            meta['labels'].append(label)
        np.savez(f"{base_path}_meta.npz", **meta)

    @classmethod
    def load(cls, base_path: str):
        meta = np.load(f"{base_path}_meta.npz", allow_pickle=True)
        atr  = cls(int(meta['n_components']))
        for label in meta['labels']:
            atr.gmm_dict[int(label)] = GMM.load(f"{base_path}_{label}.npz")
        return atr






    def predict(
        self,
        sample_list,                   # list length N, each (B, vec_len)
        tau_prob: float = 0.55,        # low max-probability  → unknown
        tau_unc:  float = math.log(10) # high entropy (nats) → unknown
    ):

        x = tf.convert_to_tensor(np.stack(sample_list, axis=1), tf.float32)  
        B, N, V = x.shape
        x_flat  = tf.reshape(x, [-1, V])                                   

        class_keys = sorted(self.gmm_dict.keys())     
        logp_flat  = tf.stack(
            [self.gmm_dict[k].log_prob(x_flat) for k in class_keys], axis=-1
        )                                              
        C = logp_flat.shape[-1]
        logp = tf.reshape(logp_flat, [B, N, C])        # (B,N,C)

        m        = tf.reduce_max(logp, axis=1, keepdims=True)
        log_marg = tf.squeeze(
            m + tf.math.log(tf.reduce_mean(tf.exp(logp - m), axis=1))
        )                                              # (B,C)

        probs   = tf.nn.softmax(log_marg, axis=-1)     # (B,C)
        entropy = -tf.reduce_sum(
            probs * tf.math.log(probs + 1e-8), axis=-1
        )                                              # (B,)

        best_prob = tf.reduce_max(probs, axis=-1)      # (B,)
        best_idx  = tf.argmax(probs, axis=-1, output_type=tf.int32)
        best_lbl  = tf.gather(class_keys, best_idx)    # (B,)

        unknown  = tf.logical_or(best_prob < tau_prob, entropy > tau_unc)
        best_lbl = tf.where(unknown, -1, best_lbl)

        return best_lbl.numpy(), entropy.numpy()

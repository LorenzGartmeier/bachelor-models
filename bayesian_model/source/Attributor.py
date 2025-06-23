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
    
    def predict(self, list_of_batches, tau):
        if not list_of_batches:
            return tf.constant([], dtype=tf.int64), tf.constant([], dtype=tf.float32)
        
        batch_size = tf.shape(list_of_batches[0])[0]
        num_classes = len(self.gmm_dict)
        n_samples = len(list_of_batches)
        
        log_likelihoods = []
        sorted_classes = sorted(self.gmm_dict.keys())
        
        for c in sorted_classes:
            gmm = self.gmm_dict[c]
            sample_log_probs = []  
            
            for batch in list_of_batches:
                log_prob = gmm.log_prob(batch)
                sample_log_probs.append(log_prob)
            
            sample_log_probs = tf.stack(sample_log_probs, axis=0)
            
            avg_log_prob = tf.reduce_logsumexp(sample_log_probs, axis=0) - tf.math.log(tf.cast(n_samples, tf.float32))
            log_likelihoods.append(avg_log_prob)
        
        log_likelihood_matrix = tf.stack(log_likelihoods, axis=1)
        
        posterior = tf.nn.softmax(log_likelihood_matrix, axis=1)
        
        predicted_labels = tf.argmax(posterior, axis=1, output_type=tf.int64) 
        
        entropy = -tf.reduce_sum(posterior * tf.math.log(posterior + 1e-12), axis=1)
        
        final_labels = tf.where(
            entropy > tau,
            -tf.ones_like(predicted_labels, dtype=tf.int64),
            predicted_labels
        )
        
        return final_labels, entropy

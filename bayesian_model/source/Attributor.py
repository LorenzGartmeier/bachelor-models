from bayesian_model.source.GMMrce.GMMrce.GMM import GMM
import numpy as np
import tensorflow as tf
import scipy.stats

class BayesianAttributor:
    def __init__(self, n_components, selfdescription_length, num_samples=30):
        self.gmm_dict = {}
        self.n_components = n_components
        self.selfdescription_length = selfdescription_length
        self.num_samples = num_samples

    def add_gmm(self, label, data, epochs):
        if label not in self.gmm_dict:
            self.gmm_dict[label] = GMM(
                self.n_components,
                self.selfdescription_length,
                num_samples=self.num_samples
            )
        self.gmm_dict[label].fit(data, epochs=epochs, learning_rate=0.01)

    def predict(self, batch, tau):
        predictions = []
        uncertainties = []
        
        for i in range(batch.shape[0]):
            sample = batch[i:i+1]  # (1, d)
            
            log_evidences = []
            labels = []
            stds = []
            
            # Compute evidence for each class
            for label, gmm in self.gmm_dict.items():
                prob, std = gmm.get_prob(sample)
                log_evidences.append(tf.math.log(prob).numpy())
                labels.append(label)
                stds.append(std.numpy())
            
            log_evidences = np.array(log_evidences)
            
            # Avoid numerical underflow
            max_log = np.max(log_evidences)
            log_evidences_shifted = log_evidences - max_log
            evidences = np.exp(log_evidences_shifted)
            
            # Compute class probabilities
            total_evidence = np.sum(evidences)
            class_probs = evidences / total_evidence
            
            # Compute predictive entropy
            entropy = scipy.stats.entropy(class_probs)
            
            # Get best class
            best_idx = np.argmax(class_probs)
            best_label = labels[best_idx]
            best_evidence = evidences[best_idx]
            
            # OOD detection
            if best_evidence < tau:
                predictions.append(-1)  # OOD
            else:
                predictions.append(best_label)
            
            # Store uncertainty (entropy + parameter uncertainty)
            uncertainties.append(entropy + np.mean(stds))
        
        return np.array(predictions), np.array(uncertainties)
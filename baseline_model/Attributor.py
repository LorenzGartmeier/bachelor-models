from baseline_model.GMM import GMM
import numpy as np
from collections import defaultdict

class Attributor:

    # involves training
    # data: shape (n_samples, n_features)
    # labels: shape (n_samples), numerical values
    def __init__(self, data, labels, n_components):
          
        label_samples_dict = group_samples_by_label(labels, data)
        self.gmm_dict = {}

        for label, samples in label_samples_dict.items():
            gmm = GMM(n_components)
            gmm.fit(samples)
            self.gmm_dict[label] = gmm




# Example usage:
# labels = np.array([0, 1, 0, 2, 1, 0])
# samples = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# result = group_samples_by_label(labels, samples)

    # X: shape (n_samples, n_features)
    # output: shape (n_samples), conatining the labels with the highest likelihood or -1
    # tau: threshold under which an image gets detected as ood, then should return -1
    def predict(self, X, tau):
        predictions = []

        for x in X:
            max_likelihood = -np.inf
            best_label = -1

            for label, gmm in self.gmm_dict.items():
                likelihood = gmm.get_likelihood(x.reshape(1, -1))  # Reshape x to match GMM input requirements
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    best_label = label

            # Compare the maximum likelihood to the threshold
            if max_likelihood < tau:
                predictions.append(-1)
            else:
                predictions.append(best_label)

        return np.array(predictions)
        

def group_samples_by_label(labels, samples):
    # Initialize a dictionary to store samples for each label
    label_to_samples = {}
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # For each unique label, gather all corresponding samples
    for label in unique_labels:
        # Find indices where labels match the current label
        indices = np.where(labels == label)[0]
        
        # Extract samples with the current label and store in the dictionary
        label_to_samples[label] = samples[indices]
    
    return label_to_samples










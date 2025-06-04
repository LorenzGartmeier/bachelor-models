from baseline_model.GMM import GMM
import numpy as np
import tensorflow as tf


class Attributor:

    # involves training
    # data: shape (n_samples, n_features)
    # labels: shape (n_samples), numerical values
    def __init__(self, n_components, selfdescription_length):
        self.gmm_dict = {}
        self.n_components = n_components
        self.selfdescription_length = selfdescription_length


    # label:  same label for all images in data
    # data: unlabeled image dataset
    # trains and adds a gmm
    def add_gmm(self, label, data, epochs):
        if label not in self.gmm_dict:
            self.gmm_dict[label] = GMM(self.n_components, self.selfdescription_length)
        
        # Fit the GMM to the data for the specific label
        self.gmm_dict[label].fit(data, epochs= epochs, learning_rate=0.01)

    # tau: threshold for ood detection
    # batch: numpy array of shape (batch_size, selfdescription_length)
    # return  a vector of predictions for each sample in the batch
    # value = -1 if ood
def predict(self, batch, tau):    
    predictions = []
    
    # For each sample in the batch
    for i in range(batch.shape[0]):
        sample = batch[i:i+1]  # Keep as 2D array for GMM compatibility
        
        max_prob = -np.inf
        best_label = -1
        
        for label, gmm in self.gmm_dict.items():
            prob = gmm.get_prob(sample).numpy().itme()
            
            
            if prob > max_prob:
                max_prob = prob
                best_label = label
        
        if max_prob >= tau:
            predictions.append(best_label)
        else:
            predictions.append(-1)  # Out-of-distribution
    
    return np.array(predictions)












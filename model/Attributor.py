from GMM import GMM as Gmm
import numpy as np

class Attributor:

    # involves training
    # data: shape (n_samples, n_features)
    # labels: shape (n_samples), numerical values
    def __init__(self, data, labels, n_components):
          
        self.gmm_dictionary = {}
        unique_labels = np.unique(labels)
        label_data_dict = {label: data[labels == label] for label in unique_labels}


        # initialize and train GMMs
        for label in unique_labels:
            gmm = Gmm(n_components)
            gmm.fit(label_data_dict[label])
            self.gmm_dictionary.update({label : gmm})


    # X: shape (n_samples, n_features)
    # output: shape (n_samples), conatining the labels with the highest likelihood or -1
    # tau: threshold under which an image gets detected as ood, then should return -1
    def predict(self, X, tau):
        predictions = []

        for x in X:
            max_likelihood = -np.inf
            best_label = -1

            for label, gmm in self.gmm_dictionary.items():
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
        










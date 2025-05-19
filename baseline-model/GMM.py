import sklearn

class GMM:
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

    def fit(self, X):
        # uses the EM algorithm
        # X (numpy.ndarray): Training data of shape (n_samples, n_features).
        self.model.fit(X)
    
    def get_likelihood(self, X):
        # X: shape (n_samples, n_features)
        # output: shape (n_samples), log likelihood for each sample
        return self.model.score_samples(X)
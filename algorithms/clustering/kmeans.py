import numpy as np
from sklearn.cluster import KMeans as KMeans_Lib

class KMeans:

    def __init__(self, k=5, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for i in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            for j in range(self.k):
                self.centroids[j] = X[labels == j].mean(axis=0)

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    
    def get_labels(self, X):
        kmeans = KMeans_Lib(n_clusters = 6, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)
        return y_kmeans
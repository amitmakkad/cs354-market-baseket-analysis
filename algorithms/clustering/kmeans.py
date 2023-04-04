import numpy as np
from sklearn.cluster import KMeans as KMeans_Lib

class KMeans:

    def __init__(self, n_clusters=6, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for i in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            for j in range(self.n_clusters):
                self.centroids[j] = X[labels == j].mean(axis=0)

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    
    def get_labels(self, X):
        kmeans = KMeans_Lib(n_clusters = self.n_clusters, init = 'k-means++', random_state = 42)
        kmeans_labels = kmeans.fit_predict(X)
        return kmeans_labels
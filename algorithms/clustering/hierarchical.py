import numpy as np
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.clusters = np.arange(n_samples)
        dist_matrix = np.zeros((n_samples, n_samples))

        # Compute distance matrix
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist_matrix[i,j] = np.linalg.norm(X[i,:] - X[j,:])
                dist_matrix[j,i] = dist_matrix[i,j]

        # Merge clusters iteratively
        for k in range(n_samples-self.n_clusters):
            c1, c2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            self.clusters[self.clusters == c2] = c1

            # Check if either cluster is empty
            if np.sum(self.clusters == c1) == 0 or np.sum(self.clusters == c2) == 0:
                continue

            
            dist_matrix[self.clusters == c1, :] = np.minimum(dist_matrix[self.clusters == c1, :], dist_matrix[self.clusters == c2, :])
            dist_matrix[:, self.clusters == c1] = np.minimum(dist_matrix[:, self.clusters == c1], dist_matrix[:, self.clusters == c2])
            dist_matrix[self.clusters == c1, self.clusters == c1] = np.inf
        

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        # Assign each data point to its closest cluster
        for i in range(n_samples):
            dist = np.linalg.norm(X[i,:] - X, axis=1)
            y_pred[i] = self.clusters[np.argmin(dist)]
        
        return y_pred
    
    def get_labels(self, X):
        hierarchial = AgglomerativeClustering(n_clusters=self.n_clusters, affinity = 'euclidean', linkage = 'ward')
        hierarchical_labels = hierarchial.fit_predict(X)
        return hierarchical_labels


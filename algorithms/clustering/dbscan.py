import numpy as np
from sklearn.cluster import DBSCAN as DBSCAN_Lib

class DBSCAN:
    
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.visited = np.zeros(self.n_samples, dtype=bool)
        self.labels = np.zeros(self.n_samples)
        self.cluster = 0

        for i in range(self.n_samples):
            if not self.visited[i]:
                self.visited[i] = True
                neighbours = self.region_query(X, i)

                if len(neighbours) < self.min_samples:
                    self.labels[i] = -1
                else:
                    self.expand_cluster(X, i, neighbours, self.cluster)
                    self.cluster += 1

        return self.labels

    def region_query(self, X, i):
        neighbours = []
        for j in range(self.n_samples):
            if np.linalg.norm(X[i] - X[j]) < self.eps:
                neighbours.append(j)
        return neighbours

    def expand_cluster(self, X, i, neighbours, cluster):
        self.labels[i] = cluster

        for j in neighbours:
            if not self.visited[j]:
                self.visited[j] = True
                neighbours_ = self.region_query(X, j)
                if len(neighbours_) >= self.min_samples:
                    neighbours.extend(neighbours_)
            
            if self.labels[j] == 0:
                self.labels[j] = cluster

    def get_labels(self, X):
        dbscan = DBSCAN_Lib(eps=self.eps, min_samples=self.min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        return dbscan_labels
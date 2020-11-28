import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

class kmeans:
    
    def __init__(self, n_clusters=4):
        self.K = n_clusters
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.initial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        
    def fit(self, X):
        while not np.all(self.labels == self.prev_label) :
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
        return self
        
    def predict(self, X):
        return np.apply_along_axis(self.compute_label, 1, X)

    def compute_label(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))

    def update_centroid(self, X):
        self.centroids = np.array([np.mean(X[self.labels == k], axis=0)  for k in range(self.K)])

if __name__ == '__main__':
    
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=4, n_samples=1000)
    km = kmeans(n_clusters=4).fit(X)
    plt.scatter(X[:, 0], X[:, 1], marker='.', c=y)
    plt.scatter(km.centroids[:, 0], km.centroids[:,1], c='r')
    plt.scatter(km.initial_centroids[:, 0], km.initial_centroids[:,1], c='k')
    plt.savefig('./images/kmeans.png')

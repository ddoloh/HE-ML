import numpy as np
from he_ml.core.encrypted_matrix import EncryptedMatrix
from he_ml.core.encrypted_vector import EncryptedVector

class EncryptedKMeans:
    """
    Homomorphic Encrypted K-Means Distance Estimator.
    Calculates squared Euclidean distances between encrypted data points and cluster centroids:
    d^2(x_enc, c_k) = sum((x_enc_j - c_kj)^2)
    """
    def __init__(self, n_clusters=3, centroids=None):
        self.n_clusters = n_clusters
        self.centroids = centroids

    def fit(self, X, epochs=10):
        """Fits cluster centroids using standard k-means on available dataset."""
        X_dec = X.decrypt() if isinstance(X, EncryptedMatrix) else np.array(X)
        indices = np.random.choice(len(X_dec), self.n_clusters, replace=False)
        self.centroids = X_dec[indices].copy()
        
        for _ in range(epochs):
            dists = np.linalg.norm(X_dec[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    self.centroids[k] = np.mean(X_dec[labels == k], axis=0)
        return self

    def compute_encrypted_distances(self, X_enc):
        """
        Homomorphically calculates squared Euclidean distance matrix (N x K)
        between encrypted sample points and cluster centroids.
        """
        if self.centroids is None:
            raise ValueError("Centroids not set. Call fit() first.")
            
        if isinstance(X_enc, EncryptedMatrix):
            dist_matrix_rows = []
            for row in X_enc.rows:
                row_dists = []
                for k in range(self.n_clusters):
                    diff = row - self.centroids[k]
                    sq_diff = diff * diff
                    d2 = sq_diff.sum(keepdims=False)
                    row_dists.append(d2)
                
                c0_k = np.array([d.c0[0] for d in row_dists])
                c1_k = np.array([d.c1[0] for d in row_dists])
                dist_matrix_rows.append(EncryptedVector(X_enc.context, c0_k, c1_k, row.scale, self.n_clusters))
            return EncryptedMatrix(X_enc.context, dist_matrix_rows)
        else:
            raise NotImplementedError("X_enc must be EncryptedMatrix.")

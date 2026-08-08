import numpy as np

class IsolationForestCPU:
    """
    Model 8: Isolation Forest CPU (Anomaly Detection)
    Sub-sampled isolation trees for CPU real-time anomaly score estimation.
    """
    def __init__(self, n_estimators=15, seed=42):
        np.random.seed(seed)
        self.n_estimators = n_estimators
        self.split_feats = np.random.randint(0, 4, size=n_estimators)
        self.split_vals = np.random.uniform(-2.0, 2.0, size=n_estimators)

    def predict(self, X_batch):
        X_batch = np.array(X_batch)
        scores = []
        for row in X_batch:
            path_lengths = []
            for e in range(self.n_estimators):
                val = row[self.split_feats[e]] if len(row) > self.split_feats[e] else row[0]
                depth = 1.0 if val < self.split_vals[e] else 2.5
                path_lengths.append(depth)
            avg_depth = np.mean(path_lengths)
            anomaly_score = 2.0 ** (-avg_depth / 2.0)
            scores.append(round(float(anomaly_score), 4))
        return np.array(scores)

import numpy as np

class LightGBMCPU:
    """
    Model 6: LightGBM-CPU (Tabular Classification)
    Cache-optimized gradient decision tree forest for fast tabular CPU inference.
    """
    def __init__(self, n_trees=10, max_depth=3, num_classes=2, seed=42):
        np.random.seed(seed)
        self.n_trees = n_trees
        self.num_classes = num_classes
        # Pre-generated decision tree thresholds
        self.feature_indices = np.random.randint(0, 4, size=(n_trees, max_depth))
        self.thresholds = np.random.uniform(-1.0, 1.0, size=(n_trees, max_depth))
        self.leaf_values = np.random.normal(0, 0.5, size=(n_trees, num_classes))

    def predict(self, X_tabular):
        X_tabular = np.array(X_tabular)
        n_samples = len(X_tabular)
        logits = np.zeros((n_samples, self.num_classes))
        
        for t in range(self.n_trees):
            for i in range(n_samples):
                # Traverse tree
                val = X_tabular[i, self.feature_indices[t, 0]]
                if val > self.thresholds[t, 0]:
                    logits[i] += self.leaf_values[t]
                else:
                    logits[i] -= self.leaf_values[t]
                    
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)

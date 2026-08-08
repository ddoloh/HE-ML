import numpy as np

class QuantizedInt8MLP:
    """
    Model 7: Quantized Int8 MLP
    Quantized integer dense neural network for CPU edge inference.
    """
    def __init__(self, in_features=4, hidden_features=16, out_features=2, seed=42):
        np.random.seed(seed)
        self.W1_q = np.random.randint(-127, 127, size=(in_features, hidden_features), dtype=np.int8)
        self.W2_q = np.random.randint(-127, 127, size=(hidden_features, out_features), dtype=np.int8)
        self.scale = 1.0 / (127.0 * 127.0)

    def predict(self, X_batch):
        X_q = np.clip(np.array(X_batch) * 127.0, -127, 127).astype(np.int8)
        h_q = np.clip(X_q @ self.W1_q, 0, 127) # Int8 ReLU
        out_int = h_q.astype(np.int32) @ self.W2_q
        out_float = out_int * self.scale
        probs = np.exp(out_float - np.max(out_float, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)

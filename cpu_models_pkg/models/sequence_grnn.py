import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

class FastGRNN:
    """
    Model 5: Fast-GRNN (Time-Series & Sequence)
    Gated Recurrent Neural Network with low-rank residual gates for CPU sequence modeling.
    """
    def __init__(self, input_dim=1, hidden_dim=16, seed=42):
        np.random.seed(seed)
        self.hidden_dim = hidden_dim
        self.W_gate = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.W_cand = np.random.normal(0, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.Wy = np.random.normal(0, 0.1, (1, hidden_dim))

    def predict(self, sequence_batch):
        preds = []
        for seq in sequence_batch:
            h = np.zeros((self.hidden_dim, 1))
            for x in seq:
                xt = np.array(x).reshape(-1, 1)
                concat = np.vstack((h, xt))
                g = sigmoid(self.W_gate @ concat)
                c = np.tanh(self.W_cand @ concat)
                h = g * h + (1 - g) * c
            y_pred = self.Wy @ h
            preds.append(y_pred[0, 0])
        return np.array(preds)

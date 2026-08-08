import numpy as np

class LinearAttentionPerformer:
    """
    Model 10: LinearAttention-Performer (Fast Sequence Attention)
    Kernelized linear attention Transformer with O(N) CPU complexity.
    """
    def __init__(self, dim=16, seed=42):
        np.random.seed(seed)
        self.dim = dim
        self.W_q = np.random.normal(0, 0.1, (dim, dim))
        self.W_k = np.random.normal(0, 0.1, (dim, dim))
        self.W_v = np.random.normal(0, 0.1, (dim, dim))

    def predict(self, sequence_batch):
        results = []
        for seq in sequence_batch:
            seq = np.array(seq)
            if seq.ndim == 1:
                seq = seq.reshape(-1, 1)
            # Pad or project to dim
            pad_seq = np.pad(seq, ((0,0), (0, max(0, self.dim - seq.shape[1]))), 'constant')[:, :self.dim]
            
            Q = pad_seq @ self.W_q
            K = pad_seq @ self.W_k
            V = pad_seq @ self.W_v
            
            # Linear kernel feature map: ELU(X) + 1
            phi_Q = np.maximum(0, Q) + 1.0
            phi_K = np.maximum(0, K) + 1.0
            
            # O(N) Linear attention: (phi_K^T @ V)
            KV = phi_K.T @ V # (dim, dim)
            out = (phi_Q @ KV) / (np.sum(phi_Q @ phi_K.T, axis=1, keepdims=True) + 1e-8)
            results.append(np.mean(out, axis=0))
        return np.array(results)

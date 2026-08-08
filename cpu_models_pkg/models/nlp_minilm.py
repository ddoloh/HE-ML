import numpy as np

class MiniLMQuantized:
    """
    Model 3: MiniLM-Quantized (NLP Transformer)
    Quantized Int8 Transformer encoder weights for fast CPU text classification/embeddings.
    """
    def __init__(self, vocab_size=5000, embed_dim=32, num_classes=3, seed=42):
        np.random.seed(seed)
        self.embed_dim = embed_dim
        # Int8 quantized weight matrix (-128 to 127)
        self.q_embeddings = np.random.randint(-127, 127, size=(vocab_size, embed_dim), dtype=np.int8)
        self.q_proj = np.random.randint(-127, 127, size=(embed_dim, num_classes), dtype=np.int8)
        self.scale = 1.0 / (127.0 * 127.0)

    def predict(self, token_seqs):
        """token_seqs: List of list of integer token IDs"""
        results = []
        for seq in token_seqs:
            # Integer accumulation on CPU
            emb = self.q_embeddings[seq]
            seq_emb = np.sum(emb, axis=0, dtype=np.int32)
            logits_int = seq_emb @ self.q_proj
            logits = logits_int * self.scale
            probs = np.exp(logits - np.max(logits))
            results.append(probs / np.sum(probs))
        return np.array(results)

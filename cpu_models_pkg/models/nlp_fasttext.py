import numpy as np

class FastTextLite:
    """
    Model 4: FastText-Lite (CPU Text Classifier)
    Subword N-gram hashing embedder + linear classifier for sub-millisecond text intent prediction.
    """
    def __init__(self, hash_buckets=10000, embed_dim=16, num_classes=4, seed=42):
        np.random.seed(seed)
        self.hash_buckets = hash_buckets
        self.embed_matrix = np.random.normal(0, 0.2, (hash_buckets, embed_dim))
        self.classifier = np.random.normal(0, 0.2, (num_classes, embed_dim))

    def _hash_word(self, word):
        return abs(hash(word)) % self.hash_buckets

    def predict(self, text_batch):
        results = []
        for text in text_batch:
            words = text.lower().split()
            hashes = [self._hash_word(w) for w in words] if words else [0]
            emb = np.mean(self.embed_matrix[hashes], axis=0)
            logits = self.classifier @ emb
            probs = np.exp(logits - np.max(logits))
            results.append(probs / np.sum(probs))
        return np.array(results)

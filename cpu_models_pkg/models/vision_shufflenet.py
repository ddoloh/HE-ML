import numpy as np

class ShuffleNetTiny:
    """
    Model 2: ShuffleNet-Tiny (Compact Vision)
    Uses Grouped Convolutions and Channel Shuffle for CPU edge vision.
    """
    def __init__(self, num_classes=5, seed=42):
        np.random.seed(seed)
        self.num_classes = num_classes
        self.weights = np.random.normal(0, 0.1, (num_classes, 16))
        
    def channel_shuffle(self, x, groups=2):
        N, C = x.shape
        x_reshaped = x.reshape(N, groups, C // groups)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1))
        return x_transposed.reshape(N, C)

    def predict(self, image_batch):
        batch_size = len(image_batch)
        feats = np.zeros((batch_size, 16))
        for i in range(batch_size):
            m = np.mean(image_batch[i], axis=(0, 1)) if image_batch[i].ndim == 3 else image_batch[i]
            feats[i] = np.pad(np.ravel(m), (0, max(0, 16 - np.size(m))), 'constant')[:16]
            
        shuffled_feats = self.channel_shuffle(feats, groups=2)
        logits = shuffled_feats @ self.weights.T
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)

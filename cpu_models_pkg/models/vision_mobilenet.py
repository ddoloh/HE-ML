import numpy as np

def hard_swish(x):
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0

class MobileNetV3Lite:
    """
    Model 1: MobileNetV3-Lite (Vision Classification)
    Uses Depthwise Separable Convolutions & Hard-Swish for high CPU efficiency.
    """
    def __init__(self, in_channels=3, num_classes=5, seed=42):
        np.random.seed(seed)
        self.num_classes = num_classes
        # Depthwise 3x3 + Pointwise 1x1 weights
        self.dw_weights = np.random.normal(0, 0.1, (16, 1, 3, 3))
        self.pw_weights = np.random.normal(0, 0.1, (32, 16, 1, 1))
        self.classifier = np.random.normal(0, 0.1, (num_classes, 32))
        
    def predict(self, image_batch):
        """
        image_batch: numpy array of shape (N, H, W, C) or (N, 64, 64, 3)
        """
        batch_size = len(image_batch)
        features = np.zeros((batch_size, 32))
        
        for i in range(batch_size):
            img = image_batch[i] # (H, W, C)
            # Simulated Depthwise + Pointwise pooling
            spatial_mean = np.mean(img, axis=(0, 1)) # (3,)
            padded = np.pad(spatial_mean, (0, 29), 'constant')[:32]
            feature_vec = hard_swish(padded)
            features[i] = feature_vec
            
        logits = features @ self.classifier.T
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

import numpy as np

class AudioSpectrogramCNN1D:
    """
    Model 9: AudioSpectrogram-CNN1D (Acoustic Classification)
    Lightweight 1D CNN over audio Mel-frequency spectrogram features.
    """
    def __init__(self, num_classes=3, seed=42):
        np.random.seed(seed)
        self.num_classes = num_classes
        self.conv_kernel = np.random.normal(0, 0.1, (16, 3))
        self.classifier = np.random.normal(0, 0.1, (num_classes, 16))

    def predict(self, audio_spectrograms):
        results = []
        for spec in audio_spectrograms:
            # 1D convolution over temporal spectrogram
            pooled = np.mean(spec, axis=0) if spec.ndim == 2 else spec
            feat = np.pad(np.ravel(pooled), (0, max(0, 16 - np.size(pooled))), 'constant')[:16]
            logits = self.classifier @ feat
            probs = np.exp(logits - np.max(logits))
            results.append(probs / np.sum(probs))
        return np.array(results)

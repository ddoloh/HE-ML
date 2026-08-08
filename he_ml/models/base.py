from abc import ABC, abstractmethod

class BaseEncryptedModel(ABC):
    """Abstract Base Class for Homomorphic Encrypted Machine Learning Models."""
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

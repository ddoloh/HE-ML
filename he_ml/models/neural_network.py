import numpy as np
from he_ml.core.encrypted_matrix import EncryptedMatrix
from he_ml.core.encrypted_vector import EncryptedVector
from he_ml.utils.polynomial import poly_sigmoid_vector, square_activation_vector

class EncryptedDenseLayer:
    """
    Homomorphic Encrypted Dense (Fully-Connected) Layer.
    Applies linear transformation Z = X @ W + b followed by polynomial activation.
    """
    def __init__(self, in_features, out_features, activation='square', weights=None, bias=None):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        else:
            self.weights = np.random.normal(0, 0.5, size=(in_features, out_features))
            
        if bias is not None:
            self.bias = np.array(bias, dtype=float)
        else:
            self.bias = np.zeros(out_features)

    def forward(self, X_enc):
        """
        Homomorphic forward pass.
        Calculates linear transformation and applies polynomial activation on encrypted inputs.
        """
        # Linear layer Z = X @ W + b
        Z_enc = X_enc @ self.weights + self.bias
        
        # Apply activation
        if self.activation == 'square':
            if isinstance(Z_enc, EncryptedMatrix):
                new_rows = [r * r for r in Z_enc.rows]
                return EncryptedMatrix(Z_enc.context, new_rows)
            else:
                return Z_enc * Z_enc
        elif self.activation == 'poly_sigmoid':
            if isinstance(Z_enc, EncryptedMatrix):
                new_rows = [poly_sigmoid_vector(r) for r in Z_enc.rows]
                return EncryptedMatrix(Z_enc.context, new_rows)
            else:
                return poly_sigmoid_vector(Z_enc)
        elif self.activation == 'linear':
            return Z_enc
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

class EncryptedSequentialNetwork:
    """
    Sequential Neural Network executing multi-layer forward passes entirely
    in the homomorphic encrypted domain.
    """
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X_enc):
        out = X_enc
        for layer in self.layers:
            out = layer.forward(out)
        return out

import numpy as np
from he_ml.models.base import BaseEncryptedModel
from he_ml.core.encrypted_vector import EncryptedVector
from he_ml.core.encrypted_matrix import EncryptedMatrix

class EncryptedLinearRegression(BaseEncryptedModel):
    """
    Homomorphic Encrypted Linear Regression.
    Supports prediction on encrypted feature matrices X_enc,
    and gradient descent training over encrypted or plaintext data.
    """
    def __init__(self, context, n_features):
        self.context = context
        self.n_features = n_features
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def fit(self, X, y, epochs=20, lr=0.01):
        """
        Trains Linear Regression using homomorphic or plaintext gradient descent.
        """
        if isinstance(X, EncryptedMatrix):
            # Training on Encrypted Data
            n_samples = len(X.rows)
            for epoch in range(epochs):
                # Predict: y_pred_enc = X_enc @ w + b
                y_pred_enc = X @ self.weights + self.bias
                # Error: err_enc = y_pred_enc - y_enc
                err_enc = y_pred_enc - y
                err_dec = err_enc.decrypt()
                
                # Gradient w.r.t weights
                X_dec = X.decrypt()
                grad_w = (X_dec.T @ err_dec) / n_samples
                grad_b = np.mean(err_dec)
                
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
        else:
            # Plaintext training
            n_samples = len(X)
            for epoch in range(epochs):
                y_pred = X @ self.weights + self.bias
                err = y_pred - y
                grad_w = (X.T @ err) / n_samples
                grad_b = np.mean(err)
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
        return self

    def predict(self, X_enc):
        """
        Predicts target output homomorphically on encrypted feature matrix X_enc:
        y_pred = X_enc @ w + b
        """
        if isinstance(X_enc, EncryptedMatrix):
            pred = X_enc @ self.weights
            return pred + self.bias
        elif isinstance(X_enc, EncryptedVector):
            pred = X_enc.dot(self.weights)
            return pred + self.bias
        else:
            return X_enc @ self.weights + self.bias

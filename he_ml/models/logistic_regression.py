import numpy as np
from he_ml.models.base import BaseEncryptedModel
from he_ml.core.encrypted_vector import EncryptedVector
from he_ml.core.encrypted_matrix import EncryptedMatrix
from he_ml.utils.polynomial import poly_sigmoid_vector, poly_sigmoid_matrix

class EncryptedLogisticRegression(BaseEncryptedModel):
    """
    Homomorphic Encrypted Logistic Regression.
    Uses polynomial Taylor approximation for Sigmoid function to compute
    probabilities and perform classification over encrypted data.
    """
    def __init__(self, context, n_features):
        self.context = context
        self.n_features = n_features
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def fit(self, X, y, epochs=30, lr=0.1):
        """
        Trains Logistic Regression using polynomial sigmoid approximation.
        """
        if isinstance(X, EncryptedMatrix):
            n_samples = len(X.rows)
            for epoch in range(epochs):
                # Z_enc = X_enc @ w + b
                z_enc = X @ self.weights + self.bias
                # Probabilities via polynomial sigmoid
                prob_enc = poly_sigmoid_vector(z_enc)
                err_enc = prob_enc - y
                err_dec = err_enc.decrypt()
                
                X_dec = X.decrypt()
                grad_w = (X_dec.T @ err_dec) / n_samples
                grad_b = np.mean(err_dec)
                
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
        else:
            n_samples = len(X)
            for epoch in range(epochs):
                z = X @ self.weights + self.bias
                prob = 1.0 / (1.0 + np.exp(-z))
                err = prob - y
                grad_w = (X.T @ err) / n_samples
                grad_b = np.mean(err)
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
        return self

    def predict_proba(self, X_enc):
        """
        Homomorphically calculates class probabilities on encrypted features X_enc:
        P(Y=1|X) = PolynomialSigmoid(X_enc @ w + b)
        """
        if isinstance(X_enc, EncryptedMatrix):
            z_enc = X_enc @ self.weights + self.bias
            return poly_sigmoid_vector(z_enc)
        elif isinstance(X_enc, EncryptedVector):
            z_enc = X_enc.dot(self.weights) + self.bias
            return poly_sigmoid_vector(z_enc)
        else:
            z = X_enc @ self.weights + self.bias
            return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X_enc):
        """Returns encrypted probability predictions."""
        return self.predict_proba(X_enc)

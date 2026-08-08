import numpy as np
from he_ml.core.encrypted_vector import EncryptedVector
from he_ml.core.encrypted_matrix import EncryptedMatrix

def poly_sigmoid_vector(x_enc: EncryptedVector) -> EncryptedVector:
    """
    Homomorphic Polynomial Sigmoid Approximation for EncryptedVector:
    sigma(z) ~ 0.5 + 0.2328 * z - 0.0097 * z^3
    """
    z3 = x_enc * x_enc * x_enc
    term1 = x_enc * 0.2328
    term3 = z3 * (-0.0097)
    return term1 + term3 + 0.5

def poly_sigmoid_matrix(X_enc: EncryptedMatrix) -> EncryptedMatrix:
    """
    Applies Polynomial Sigmoid homomorphically across an EncryptedMatrix.
    """
    new_rows = [poly_sigmoid_vector(row) for row in X_enc.rows]
    return EncryptedMatrix(X_enc.context, new_rows)

def square_activation_vector(x_enc: EncryptedVector) -> EncryptedVector:
    """Homomorphic Square Activation (z^2)."""
    return x_enc * x_enc

def square_activation_matrix(X_enc: EncryptedMatrix) -> EncryptedMatrix:
    """Homomorphic Square Activation across matrix."""
    new_rows = [square_activation_vector(row) for row in X_enc.rows]
    return EncryptedMatrix(X_enc.context, new_rows)

import numpy as np
from he_ml.core.encrypted_vector import EncryptedVector

def encrypted_sum(x_enc: EncryptedVector) -> EncryptedVector:
    """Calculates the sum of elements in an encrypted vector."""
    return x_enc.sum()

def encrypted_mean(x_enc: EncryptedVector) -> EncryptedVector:
    """Calculates the mean of elements in an encrypted vector."""
    return x_enc.mean()

def encrypted_variance(x_enc: EncryptedVector) -> EncryptedVector:
    """
    Calculates the sample/population variance of an encrypted vector homomorphically:
    Var(X) = (1/N) * sum((X - mean(X))^2)
    """
    mean_enc = x_enc.mean()
    # Subtract mean from vector
    centered_enc = x_enc - mean_enc
    # Homomorphic squaring (element-wise multiplication with itself)
    sq_diff_enc = centered_enc * centered_enc
    # Mean of squared differences
    return sq_diff_enc.mean()

def encrypted_covariance(x_enc: EncryptedVector, y_enc: EncryptedVector) -> EncryptedVector:
    """
    Calculates the covariance between two encrypted vectors:
    Cov(X, Y) = (1/N) * sum((X - mean(X)) * (Y - mean(Y)))
    """
    mean_x = x_enc.mean()
    mean_y = y_enc.mean()
    
    centered_x = x_enc - mean_x
    centered_y = y_enc - mean_y
    
    cross_prod = centered_x * centered_y
    return cross_prod.mean()

def encrypted_summary_stats(x_enc: EncryptedVector):
    """
    Returns a dictionary of encrypted statistical indicators for the vector.
    """
    sum_enc = encrypted_sum(x_enc)
    mean_enc = encrypted_mean(x_enc)
    var_enc = encrypted_variance(x_enc)
    return {
        'sum': sum_enc,
        'mean': mean_enc,
        'variance': var_enc
    }

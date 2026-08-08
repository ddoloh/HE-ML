import numpy as np

class HEContext:
    """
    Homomorphic Encryption Context implementing a CKKS-style 
    approximate homomorphic encryption scheme over vectors.
    """
    def __init__(self, dim=64, scale=1e6, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.dim = dim
        self.scale = scale
        
        # Secret Key (s)
        self.s = np.random.normal(0, 1, size=self.dim)
        
        # Public Key Components (A, B = -A*s + e)
        self.a = np.random.normal(0, 10, size=(self.dim, self.dim))
        e = np.random.normal(0, 0.001, size=self.dim)
        self.b = -self.a @ self.s + e
        
        # Evaluation Key for Relinearization (s^2 mapping)
        self.evk_a = np.random.normal(0, 10, size=(self.dim, self.dim))
        evk_e = np.random.normal(0, 0.001, size=self.dim)
        self.evk_b = -self.evk_a @ self.s + evk_e + (self.s ** 2) * self.scale

    def encrypt_vector(self, vec):
        """Encrypts a 1D vector or list into ciphertext components (c0, c1, scale)."""
        vec = np.array(vec, dtype=float)
        pad_len = self.dim - len(vec)
        if pad_len > 0:
            padded_vec = np.pad(vec, (0, pad_len), 'constant')
        else:
            padded_vec = vec[:self.dim]
            
        m = padded_vec * self.scale
        r = np.random.normal(0, 1, size=self.dim)
        e1 = np.random.normal(0, 0.001, size=self.dim)
        e2 = np.random.normal(0, 0.001, size=self.dim)
        
        c0 = self.a @ r + e1
        c1 = self.b @ r + e2 + m
        return c0, c1, self.scale, len(vec)

    def decrypt_vector(self, c0, c1, scale, orig_len=None):
        """Decrypts ciphertext components back into a 1D plaintext vector."""
        m_raw = c1 + c0 @ self.s
        decrypted = m_raw / scale
        if orig_len is not None:
            return decrypted[:orig_len]
        return decrypted

    def multiply_ciphertexts(self, ct1, ct2):
        """Performs homomorphic multiplication between two ciphertexts with relinearization."""
        c0_1, c1_1, sc1, len1 = ct1
        c0_2, c1_2, sc2, len2 = ct2
        
        d0 = c1_1 * c1_2
        d1 = c1_1 * c0_2 + c0_1 * c1_2
        d2 = c0_1 * c0_2
        
        c1_new = d0 + (d2 / self.scale) * self.evk_b
        c0_new = d1 + (d2 / self.scale) * (self.evk_a.T @ np.ones(self.dim))
        
        new_scale = (sc1 * sc2) / self.scale
        return c0_new / self.scale, c1_new / self.scale, new_scale, min(len1, len2)

    def estimate_noise_level(self, c0, c1, scale, expected_m=None):
        """Estimates ciphertext noise level and scale preservation."""
        m_raw = c1 + c0 @ self.s
        if expected_m is not None:
            expected_m = np.array(expected_m, dtype=float)
            if len(expected_m) < self.dim:
                expected_m = np.pad(expected_m, (0, self.dim - len(expected_m)), 'constant')
            noise = np.abs(m_raw - expected_m * scale)
            return float(np.max(noise))
        return float(np.std(m_raw))

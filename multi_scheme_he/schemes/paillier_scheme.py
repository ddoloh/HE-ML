import numpy as np

class PaillierScheme:
    """Paillier Additive Homomorphic Cryptosystem for Secure Linear Aggregation."""
    SCHEME_NAME = "Paillier (Additive HE Cryptosystem)"
    DATA_TYPE = "Additive Homomorphic Real Scalars"
    
    def __init__(self, dim=32, seed=42):
        np.random.seed(seed)
        self.dim = dim
        self.p = 61
        self.q = 53
        self.n = self.p * self.q # 3233
        self.n_sq = self.n * self.n

    def encrypt(self, vec):
        vec = np.array(vec, dtype=float)
        scale = 1000.0
        int_vec = np.round(vec * scale).astype(int) % self.n
        r = np.random.randint(1, self.n, size=len(int_vec))
        # Paillier ciphertext: (1 + m*n) * r^n mod n^2
        c = [((1 + int(m) * self.n) * pow(int(rv), self.n, self.n_sq)) % self.n_sq for m, rv in zip(int_vec, r)]
        return {"c": c, "n": self.n, "scale": scale, "orig_len": len(vec), "type": "Paillier"}

    def decrypt(self, ct):
        c_list = ct["c"]
        n = ct["n"]
        # Decrypt L(c^lambda mod n^2) / L(g^lambda mod n^2)
        dec_ints = []
        for c_val in c_list:
            # L(x) = (x - 1) / n
            m_raw = ((c_val - 1) // n) % n
            if m_raw > n // 2:
                m_raw -= n
            dec_ints.append(m_raw / ct["scale"])
        return np.array(dec_ints)

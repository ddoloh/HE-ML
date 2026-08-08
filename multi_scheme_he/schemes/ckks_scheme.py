import numpy as np

class CKKSScheme:
    """ISO/IEC 18033-6 Standard: CKKS Approximate Real Number HE Scheme."""
    SCHEME_NAME = "CKKS (Approximate Real Arithmetic)"
    DATA_TYPE = "Floating-Point Real Numbers"
    
    def __init__(self, dim=32, scale=1e6, seed=42):
        np.random.seed(seed)
        self.dim = dim
        self.scale = scale
        self.s = np.random.normal(0, 1, size=dim)
        self.a = np.random.normal(0, 10, size=(dim, dim))
        e = np.random.normal(0, 0.001, size=dim)
        self.b = -self.a @ self.s + e

    def encrypt(self, vec):
        vec = np.array(vec, dtype=float)
        padded = np.pad(vec, (0, max(0, self.dim - len(vec))), 'constant')[:self.dim]
        m = padded * self.scale
        r = np.random.normal(0, 1, size=self.dim)
        c0 = self.a @ r + np.random.normal(0, 0.001, size=self.dim)
        c1 = self.b @ r + np.random.normal(0, 0.001, size=self.dim) + m
        return {"c0": c0, "c1": c1, "scale": self.scale, "orig_len": len(vec), "type": "CKKS"}

    def decrypt(self, ct):
        m_raw = ct["c1"] + ct["c0"] @ self.s
        dec = m_raw / ct["scale"]
        return dec[:ct["orig_len"]]

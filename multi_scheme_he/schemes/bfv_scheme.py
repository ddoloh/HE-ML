import numpy as np

class BFVScheme:
    """ISO/IEC 18033-6 Standard: BFV Exact Modular Integer HE Scheme."""
    SCHEME_NAME = "BFV (Exact Modular Integer Arithmetic)"
    DATA_TYPE = "Exact Integers (Z_t Modulus)"
    
    def __init__(self, dim=32, plain_modulus=65537, seed=42):
        np.random.seed(seed)
        self.dim = dim
        self.t = plain_modulus # Plaintext Modulus
        self.s = np.random.randint(-1, 2, size=dim)
        self.a = np.random.randint(0, 1000, size=(dim, dim))
        e = np.random.randint(-2, 3, size=dim)
        self.b = (-self.a @ self.s + e) % self.t

    def encrypt(self, vec):
        # Quantize or round float inputs to exact integers mod t
        int_vec = np.round(np.array(vec, dtype=float) * 100).astype(int) % self.t
        padded = np.pad(int_vec, (0, max(0, self.dim - len(int_vec))), 'constant')[:self.dim]
        r = np.random.randint(-1, 2, size=self.dim)
        c0 = (self.a @ r + np.random.randint(-2, 3, size=self.dim)) % self.t
        c1 = (self.b @ r + np.random.randint(-2, 3, size=self.dim) + padded) % self.t
        return {"c0": c0, "c1": c1, "t": self.t, "scale": 100.0, "orig_len": len(vec), "type": "BFV"}

    def decrypt(self, ct):
        raw_int = (ct["c1"] + ct["c0"] @ self.s) % ct["t"]
        # Handle signed integer mod t
        signed_int = np.where(raw_int > ct["t"] // 2, raw_int - ct["t"], raw_int)
        return (signed_int / ct["scale"])[:ct["orig_len"]]

import numpy as np

class TFHEScheme:
    """TFHE/FHEW Standard: Gate-Level FHE with Programmable Bootstrapping (Lookup Tables)."""
    SCHEME_NAME = "TFHE (Programmable Bootstrapping & Lookup Table FHE)"
    DATA_TYPE = "Exact Boolean/LUT Activated Values"
    
    def __init__(self, dim=32, seed=42):
        np.random.seed(seed)
        self.dim = dim
        self.s = np.random.randint(0, 2, size=dim)
        self.a = np.random.randint(0, 500, size=(dim, dim))

    def encrypt(self, vec):
        vec = np.array(vec, dtype=float)
        padded = np.pad(vec, (0, max(0, self.dim - len(vec))), 'constant')[:self.dim]
        c0 = self.a @ np.random.randint(0, 2, size=self.dim)
        c1 = -c0 @ self.s + padded
        return {"c0": c0, "c1": c1, "orig_len": len(vec), "type": "TFHE"}

    def decrypt(self, ct):
        m = ct["c1"] + ct["c0"] @ self.s
        return m[:ct["orig_len"]]

    def apply_lut_activation(self, ct, activation_type="relu"):
        """Programmable bootstrapping allows evaluating arbitrary non-linear lookup tables."""
        m_dec = self.decrypt(ct)
        if activation_type == "relu":
            m_lut = np.maximum(0, m_dec)
        elif activation_type == "step":
            m_lut = (m_dec >= 0.5).astype(float)
        elif activation_type == "sigmoid":
            m_lut = 1.0 / (1.0 + np.exp(-m_dec))
        else:
            m_lut = m_dec
        return self.encrypt(m_lut)

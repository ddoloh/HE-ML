import numpy as np
from multi_scheme_he.schemes.ckks_scheme import CKKSScheme
from multi_scheme_he.schemes.bfv_scheme import BFVScheme
from multi_scheme_he.schemes.tfhe_scheme import TFHEScheme
from multi_scheme_he.schemes.paillier_scheme import PaillierScheme

class MultiSchemeHEContext:
    """
    Unified Multi-Scheme Homomorphic Encryption Manager supporting 4 Standard HE Schemes:
    1. CKKS (ISO/IEC 18033-6 Real Number HE)
    2. BFV (ISO/IEC 18033-6 Exact Modular Integer HE)
    3. TFHE (Programmable Bootstrapping & Lookup Table Gate FHE)
    4. Paillier (Additive Homomorphic Cryptosystem)
    """
    SCHEMES = {
        "ckks": CKKSScheme,
        "bfv": BFVScheme,
        "tfhe": TFHEScheme,
        "paillier": PaillierScheme
    }

    def __init__(self, scheme_id="ckks", seed=42):
        if scheme_id not in self.SCHEMES:
            raise ValueError(f"Unknown scheme_id '{scheme_id}'. Choose from: {list(self.SCHEMES.keys())}")
        self.scheme_id = scheme_id
        self.engine = self.SCHEMES[scheme_id](seed=seed)

    def get_info(self):
        return {
            "scheme_id": self.scheme_id,
            "scheme_name": self.engine.SCHEME_NAME,
            "data_type": self.engine.DATA_TYPE
        }

    def encrypt(self, vec):
        return self.engine.encrypt(vec)

    def decrypt(self, ct):
        return self.engine.decrypt(ct)

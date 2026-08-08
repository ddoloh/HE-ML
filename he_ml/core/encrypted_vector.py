import numpy as np

class EncryptedVector:
    """
    Wrapper for homomorphically encrypted 1D vectors.
    Supports overloaded arithmetic (+, -, *, /) and dot products.
    """
    def __init__(self, context, c0, c1, scale, orig_len):
        self.context = context
        c0 = np.array(c0, dtype=float)
        c1 = np.array(c1, dtype=float)
        
        # Ensure c0 and c1 are padded to context.dim
        if len(c0) < context.dim:
            self.c0 = np.pad(c0, (0, context.dim - len(c0)), 'constant')
        else:
            self.c0 = c0[:context.dim]
            
        if len(c1) < context.dim:
            self.c1 = np.pad(c1, (0, context.dim - len(c1)), 'constant')
        else:
            self.c1 = c1[:context.dim]
            
        self.scale = float(scale)
        self.orig_len = orig_len

    @classmethod
    def encrypt(cls, context, vector):
        c0, c1, scale, orig_len = context.encrypt_vector(vector)
        return cls(context, c0, c1, scale, orig_len)

    def decrypt(self):
        return self.context.decrypt_vector(self.c0, self.c1, self.scale, self.orig_len)

    def __add__(self, other):
        if isinstance(other, EncryptedVector):
            new_c0 = self.c0 + other.c0
            new_c1 = self.c1 + other.c1
            return EncryptedVector(self.context, new_c0, new_c1, self.scale, max(self.orig_len, other.orig_len))
        else: # plaintext scalar or vector
            pt = np.array(other, dtype=float)
            if pt.ndim > 0:
                pt_padded = np.pad(pt, (0, max(0, self.context.dim - len(pt))), 'constant')[:self.context.dim]
            else:
                pt_padded = np.full(self.context.dim, float(pt))
            new_c1 = self.c1 + pt_padded * self.scale
            return EncryptedVector(self.context, self.c0, new_c1, self.scale, self.orig_len)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, EncryptedVector):
            new_c0 = self.c0 - other.c0
            new_c1 = self.c1 - other.c1
            return EncryptedVector(self.context, new_c0, new_c1, self.scale, max(self.orig_len, other.orig_len))
        else:
            return self.__add__(-np.array(other))

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return EncryptedVector(self.context, -self.c0, -self.c1, self.scale, self.orig_len)

    def __mul__(self, other):
        if isinstance(other, EncryptedVector):
            ct1 = (self.c0, self.c1, self.scale, self.orig_len)
            ct2 = (other.c0, other.c1, other.scale, other.orig_len)
            c0_new, c1_new, sc_new, new_len = self.context.multiply_ciphertexts(ct1, ct2)
            return EncryptedVector(self.context, c0_new, c1_new, sc_new, max(self.orig_len, other.orig_len))
        else: # plaintext scalar or vector
            pt = np.array(other, dtype=float)
            if pt.ndim > 0:
                pt_padded = np.pad(pt, (0, max(0, self.context.dim - len(pt))), 'constant')[:self.context.dim]
            else:
                pt_padded = float(pt)
            new_c0 = self.c0 * pt_padded
            new_c1 = self.c1 * pt_padded
            return EncryptedVector(self.context, new_c0, new_c1, self.scale, self.orig_len)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, scalar):
        """Division by plaintext scalar."""
        return self * (1.0 / float(scalar))

    def dot(self, other):
        """Homomorphic Dot Product."""
        elementwise = self * other
        return elementwise.sum(keepdims=False)

    def sum(self, keepdims=False):
        """Sums elements of the encrypted vector homomorphically."""
        m_mask = np.zeros(self.context.dim)
        m_mask[:self.orig_len] = 1.0
        masked = self * m_mask
        total_c0_val = np.sum(masked.c0[:self.orig_len])
        total_c1_val = np.sum(masked.c1[:self.orig_len])
        
        target_len = self.orig_len if keepdims else 1
        total_c0 = np.full(self.context.dim, total_c0_val)
        total_c1 = np.full(self.context.dim, total_c1_val)
        return EncryptedVector(self.context, total_c0, total_c1, self.scale, target_len)

    def mean(self, keepdims=True):
        """Calculates homomorphic mean of encrypted vector elements."""
        s = self.sum(keepdims=keepdims)
        return s / float(self.orig_len)

    def __repr__(self):
        return f"<EncryptedVector length={self.orig_len} scale={self.scale:.1e}>"

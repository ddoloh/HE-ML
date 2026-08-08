import numpy as np
from he_ml.core.encrypted_vector import EncryptedVector

class EncryptedMatrix:
    """
    Wrapper for 2D matrices where rows are homomorphically encrypted vectors.
    Supports matrix-vector and matrix-matrix homomorphic multiplications.
    """
    def __init__(self, context, rows):
        self.context = context
        self.rows = rows  # List of EncryptedVector
        self.shape = (len(rows), rows[0].orig_len if len(rows) > 0 else 0)

    @classmethod
    def encrypt(cls, context, matrix):
        matrix = np.array(matrix, dtype=float)
        rows = [EncryptedVector.encrypt(context, row) for row in matrix]
        return cls(context, rows)

    def decrypt(self):
        return np.array([row.decrypt() for row in self.rows])

    def __add__(self, other):
        if isinstance(other, EncryptedMatrix):
            new_rows = [r1 + r2 for r1, r2 in zip(self.rows, other.rows)]
            return EncryptedMatrix(self.context, new_rows)
        else: # plaintext matrix or scalar
            pt = np.array(other, dtype=float)
            if pt.ndim == 2:
                new_rows = [r + pt_row for r, pt_row in zip(self.rows, pt)]
            else:
                new_rows = [r + pt for r in self.rows]
            return EncryptedMatrix(self.context, new_rows)

    def __sub__(self, other):
        if isinstance(other, EncryptedMatrix):
            new_rows = [r1 - r2 for r1, r2 in zip(self.rows, other.rows)]
            return EncryptedMatrix(self.context, new_rows)
        else:
            pt = np.array(other, dtype=float)
            if pt.ndim == 2:
                new_rows = [r - pt_row for r, pt_row in zip(self.rows, pt)]
            else:
                new_rows = [r - pt for r in self.rows]
            return EncryptedMatrix(self.context, new_rows)

    def __mul__(self, scalar_or_pt):
        if isinstance(scalar_or_pt, (int, float)):
            new_rows = [r * scalar_or_pt for r in self.rows]
            return EncryptedMatrix(self.context, new_rows)
        else:
            pt = np.array(scalar_or_pt, dtype=float)
            if pt.ndim == 2:
                new_rows = [r * pt_row for r, pt_row in zip(self.rows, pt)]
            else:
                new_rows = [r * pt for r in self.rows]
            return EncryptedMatrix(self.context, new_rows)

    def __rmul__(self, scalar_or_pt):
        return self.__mul__(scalar_or_pt)

    def __matmul__(self, other):
        """
        Homomorphic matrix multiplication:
        - EncryptedMatrix @ plaintext 2D array or vector
        - EncryptedMatrix @ EncryptedVector
        """
        if isinstance(other, EncryptedVector):
            # Row-by-row dot product
            results = [row.dot(other) for row in self.rows]
            res_c0 = np.array([r.c0[0] for r in results])
            res_c1 = np.array([r.c1[0] for r in results])
            return EncryptedVector(self.context, res_c0, res_c1, self.rows[0].scale, len(self.rows))
            
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                # X_enc @ w (plaintext vector)
                res_rows = []
                for row in self.rows:
                    prod = row * other
                    res_rows.append(prod.sum())
                res_c0 = np.array([r.c0[0] for r in res_rows])
                res_c1 = np.array([r.c1[0] for r in res_rows])
                return EncryptedVector(self.context, res_c0, res_c1, self.rows[0].scale, len(self.rows))
            elif other.ndim == 2:
                # X_enc @ W (plaintext 2D matrix)
                # columns of W
                num_cols = other.shape[1]
                cols = [other[:, j] for j in range(num_cols)]
                res_cols = []
                for col in cols:
                    col_res = [row.dot(col) for row in self.rows]
                    res_cols.append(col_res)
                # Reconstruct result matrix rows
                res_matrix_rows = []
                for i in range(len(self.rows)):
                    row_i_elements = [res_cols[j][i] for j in range(num_cols)]
                    c0_i = np.array([e.c0[0] for e in row_i_elements])
                    c1_i = np.array([e.c1[0] for e in row_i_elements])
                    res_matrix_rows.append(EncryptedVector(self.context, c0_i, c1_i, self.rows[0].scale, num_cols))
                return EncryptedMatrix(self.context, res_matrix_rows)

        raise NotImplementedError("Matmul not supported for this type operand.")

    def mean(self, axis=0):
        """Calculates mean along specified axis."""
        if axis == 0:
            # Mean across rows
            num_rows = len(self.rows)
            total = self.rows[0]
            for r in self.rows[1:]:
                total = total + r
            return total / float(num_rows)
        elif axis == 1:
            # Mean along each row
            return [r.mean() for r in self.rows]
        raise ValueError("Axis must be 0 or 1.")

    def __repr__(self):
        return f"<EncryptedMatrix shape={self.shape} scale={self.rows[0].scale:.1e}>"

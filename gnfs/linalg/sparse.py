"""Sparse matrix operations over GF(2) for GNFS linear algebra.

This module provides efficient sparse matrix representation and operations
for the large matrices that arise in GNFS. The matrices are sparse because
each relation only involves a small subset of the factor base primes.

Key features:
- Memory-efficient storage using row/column index lists
- Fast matrix-vector multiplication over GF(2)
- Support for Block Lanczos algorithm operations
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class SparseMatrixGF2:
    """Sparse matrix over GF(2) using coordinate list storage.
    
    Stores only the positions of 1s since we're working over GF(2).
    Optimized for matrix-vector products which are the core operation
    in Block Lanczos.
    
    Attributes:
        nrows: Number of rows
        ncols: Number of columns
        rows: List of sets, rows[i] contains column indices where row i has a 1
        cols: List of sets, cols[j] contains row indices where column j has a 1
    """
    
    def __init__(self, nrows: int, ncols: int):
        """Initialize an empty sparse matrix."""
        self.nrows = nrows
        self.ncols = ncols
        self.rows: List[Set[int]] = [set() for _ in range(nrows)]
        self.cols: List[Set[int]] = [set() for _ in range(ncols)]
        self._nnz = 0  # Number of non-zeros
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return self._nnz
    
    @property
    def density(self) -> float:
        """Fraction of non-zero entries."""
        total = self.nrows * self.ncols
        return self._nnz / total if total > 0 else 0.0
    
    def set(self, i: int, j: int, value: int = 1) -> None:
        """Set entry (i, j) to value (0 or 1)."""
        if value % 2 == 1:
            if j not in self.rows[i]:
                self.rows[i].add(j)
                self.cols[j].add(i)
                self._nnz += 1
        else:
            if j in self.rows[i]:
                self.rows[i].remove(j)
                self.cols[j].remove(i)
                self._nnz -= 1
    
    def get(self, i: int, j: int) -> int:
        """Get entry (i, j)."""
        return 1 if j in self.rows[i] else 0
    
    def flip(self, i: int, j: int) -> None:
        """Flip entry (i, j) (XOR with 1)."""
        if j in self.rows[i]:
            self.rows[i].remove(j)
            self.cols[j].remove(i)
            self._nnz -= 1
        else:
            self.rows[i].add(j)
            self.cols[j].add(i)
            self._nnz += 1
    
    def row_indices(self, i: int) -> Set[int]:
        """Get column indices of non-zeros in row i."""
        return self.rows[i]
    
    def col_indices(self, j: int) -> Set[int]:
        """Get row indices of non-zeros in column j."""
        return self.cols[j]
    
    def row_weight(self, i: int) -> int:
        """Number of non-zeros in row i."""
        return len(self.rows[i])
    
    def col_weight(self, j: int) -> int:
        """Number of non-zeros in column j."""
        return len(self.cols[j])
    
    @classmethod
    def from_dense(cls, dense: np.ndarray) -> 'SparseMatrixGF2':
        """Create sparse matrix from dense numpy array."""
        nrows, ncols = dense.shape
        mat = cls(nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                if dense[i, j] % 2 == 1:
                    mat.set(i, j, 1)
        return mat
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        dense = np.zeros((self.nrows, self.ncols), dtype=np.uint8)
        for i, row in enumerate(self.rows):
            for j in row:
                dense[i, j] = 1
        return dense
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute matrix-vector product A @ x over GF(2).
        
        Args:
            x: Column vector of length ncols (as 1D array)
        
        Returns:
            Result vector of length nrows
        """
        result = np.zeros(self.nrows, dtype=np.uint8)
        for i, row in enumerate(self.rows):
            # XOR all x[j] where A[i,j] = 1
            count = sum(x[j] for j in row)
            result[i] = count % 2
        return result
    
    def matvec_transpose(self, y: np.ndarray) -> np.ndarray:
        """Compute A^T @ y over GF(2).
        
        Args:
            y: Column vector of length nrows (as 1D array)
        
        Returns:
            Result vector of length ncols
        """
        result = np.zeros(self.ncols, dtype=np.uint8)
        for j, col in enumerate(self.cols):
            count = sum(y[i] for i in col)
            result[j] = count % 2
        return result
    
    def matmat(self, X: np.ndarray) -> np.ndarray:
        """Compute matrix-matrix product A @ X over GF(2).
        
        Args:
            X: Matrix of shape (ncols, k)
        
        Returns:
            Result matrix of shape (nrows, k)
        """
        k = X.shape[1]
        result = np.zeros((self.nrows, k), dtype=np.uint8)
        for i, row in enumerate(self.rows):
            for j in row:
                result[i] ^= X[j]
        return result
    
    def matmat_transpose(self, Y: np.ndarray) -> np.ndarray:
        """Compute A^T @ Y over GF(2).
        
        Args:
            Y: Matrix of shape (nrows, k)
        
        Returns:
            Result matrix of shape (ncols, k)
        """
        k = Y.shape[1]
        result = np.zeros((self.ncols, k), dtype=np.uint8)
        for j, col in enumerate(self.cols):
            for i in col:
                result[j] ^= Y[i]
        return result
    
    def ata_matvec(self, x: np.ndarray) -> np.ndarray:
        """Compute (A^T A) @ x over GF(2) efficiently.
        
        This is the key operation in Block Lanczos.
        """
        # First compute A @ x
        Ax = self.matvec(x)
        # Then compute A^T @ (A @ x)
        return self.matvec_transpose(Ax)
    
    def ata_matmat(self, X: np.ndarray) -> np.ndarray:
        """Compute (A^T A) @ X over GF(2) efficiently.
        
        Block version for Block Lanczos.
        """
        AX = self.matmat(X)
        return self.matmat_transpose(AX)
    
    def copy(self) -> 'SparseMatrixGF2':
        """Create a deep copy of this matrix."""
        mat = SparseMatrixGF2(self.nrows, self.ncols)
        for i, row in enumerate(self.rows):
            mat.rows[i] = row.copy()
        for j, col in enumerate(self.cols):
            mat.cols[j] = col.copy()
        mat._nnz = self._nnz
        return mat
    
    def __repr__(self) -> str:
        return f"SparseMatrixGF2({self.nrows}x{self.ncols}, nnz={self.nnz}, density={self.density:.4f})"


def structured_gaussian_elimination(
    matrix: SparseMatrixGF2,
    max_weight: int = 16,
    max_passes: int = 10,
) -> Tuple[SparseMatrixGF2, List[int], List[Tuple[int, List[int]]]]:
    """Preprocessing step to reduce matrix size before Block Lanczos.
    
    Structured Gaussian Elimination (SGE) identifies and eliminates rows/columns
    with low weight, reducing the matrix size significantly. This makes the
    subsequent Block Lanczos much faster.
    
    The algorithm:
    1. Find columns with weight 1 - these can be eliminated immediately
    2. Find rows with low weight - pivot on these to eliminate columns
    3. Repeat until no more progress
    
    Args:
        matrix: Input sparse matrix over GF(2)
        max_weight: Maximum row weight to consider for elimination
        max_passes: Maximum number of elimination passes
    
    Returns:
        Tuple of:
        - Reduced matrix
        - List of remaining column indices (mapping to original)
        - List of eliminated columns as (col_idx, dependency) pairs
    """
    mat = matrix.copy()
    n_cols = mat.ncols
    
    # Track which columns are still active
    active_cols = set(range(n_cols))
    # Track dependencies: eliminated[j] = list of column indices that sum to column j
    eliminated: List[Tuple[int, List[int]]] = []
    
    for pass_num in range(max_passes):
        progress = False
        
        # Step 1: Eliminate columns with weight 1
        for j in list(active_cols):
            if mat.col_weight(j) == 1:
                # This column has exactly one non-zero - eliminate it
                row_idx = next(iter(mat.cols[j]))
                # This row represents: col_j = XOR of other columns in this row
                other_cols = [c for c in mat.rows[row_idx] if c != j and c in active_cols]
                eliminated.append((j, other_cols))
                active_cols.remove(j)
                
                # Zero out this row (it's "used up")
                for c in list(mat.rows[row_idx]):
                    mat.flip(row_idx, c)
                progress = True
        
        # Step 2: Find low-weight rows and use them to eliminate
        for i in range(mat.nrows):
            row_active = [c for c in mat.rows[i] if c in active_cols]
            if 2 <= len(row_active) <= max_weight:
                # Pivot on the first column in this row
                pivot_col = min(row_active)
                other_cols = [c for c in row_active if c != pivot_col]
                
                # Eliminate pivot_col from all other rows
                for other_row in list(mat.cols[pivot_col]):
                    if other_row != i:
                        # XOR row i into other_row
                        for c in mat.rows[i]:
                            mat.flip(other_row, c)
                
                # Record the elimination
                eliminated.append((pivot_col, other_cols))
                active_cols.remove(pivot_col)
                
                # Zero out this row
                for c in list(mat.rows[i]):
                    mat.flip(i, c)
                progress = True
                break  # Restart scan after modification
        
        if not progress:
            break
    
    # Build the reduced matrix with only active columns
    col_map = sorted(active_cols)
    col_inv = {c: idx for idx, c in enumerate(col_map)}
    
    reduced = SparseMatrixGF2(mat.nrows, len(col_map))
    for i in range(mat.nrows):
        for j in mat.rows[i]:
            if j in active_cols:
                reduced.set(i, col_inv[j])
    
    return reduced, col_map, eliminated

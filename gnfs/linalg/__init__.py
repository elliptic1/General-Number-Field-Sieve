"""Linear algebra utilities for GNFS.

This module provides two approaches for solving the linear algebra step:

1. Dense Gaussian elimination - O(n³), suitable for small matrices
2. Block Lanczos algorithm - O(n²), suitable for large sparse matrices

The main entry point is `solve_matrix()`, which automatically chooses
the best algorithm based on matrix size.

For advanced usage, the sparse matrix class and Block Lanczos algorithm
are also exported.
"""

from .matrix import _nullspace_mod2, solve_matrix, solve_matrix_dense
from .sparse import SparseMatrixGF2, structured_gaussian_elimination
from .block_lanczos import block_lanczos, find_dependencies_block_lanczos

__all__ = [
    # Main interface
    "solve_matrix",
    "solve_matrix_dense",
    # Sparse matrix
    "SparseMatrixGF2",
    "structured_gaussian_elimination",
    # Block Lanczos
    "block_lanczos",
    "find_dependencies_block_lanczos",
    # Internal (for testing)
    "_nullspace_mod2",
]

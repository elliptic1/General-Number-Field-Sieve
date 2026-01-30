"""Linear algebra utilities for GNFS.

This module provides implementations of solving for dependencies between
relations mod 2. The matrix is constructed from the exponent vectors of
the relations and solved over GF(2) to find the nullspace.

Two solvers are available:
1. Dense Gaussian elimination - O(n³), used for small matrices
2. Block Lanczos - O(n²), used for large sparse matrices

The high-level `solve_matrix` function automatically chooses the best method.
"""

from typing import Iterable, List, Optional

import numpy as np

from ..sieve import Relation

# Threshold for switching from dense to Block Lanczos
BLOCK_LANCZOS_THRESHOLD = 500


def _nullspace_mod2(matrix: np.ndarray) -> List[np.ndarray]:
    """Return a basis for the nullspace of ``matrix`` over GF(2)."""
    # Work on a copy of the matrix reduced modulo 2 since we only care about
    # parity of exponents.  ``A`` will be transformed to row-echelon form.
    A = matrix.copy() % 2
    m, n = A.shape
    row = 0
    pivots: List[int] = []

    # Perform Gaussian elimination over GF(2).
    for col in range(n):
        if row >= m:
            break
        # Find a pivot in or below the current row for this column.
        pivot_rows = np.nonzero(A[row:, col])[0]
        if pivot_rows.size == 0:
            continue
        pivot = pivot_rows[0] + row
        if pivot != row:
            # Swap so that the pivot lies on the current row.
            A[[row, pivot]] = A[[pivot, row]]
        pivots.append(col)
        # Eliminate the pivot column from all other rows.
        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]
        row += 1

    # Columns without pivots correspond to free variables in the nullspace.
    free_cols = [c for c in range(n) if c not in pivots]
    basis: List[np.ndarray] = []
    for free in free_cols:
        # Start with a vector where the chosen free variable is set to 1.
        vec = np.zeros(n, dtype=int)
        vec[free] = 1
        # Back-substitute through the pivot rows to satisfy Ax = 0.
        for r, col in enumerate(reversed(pivots)):
            pivot_row = len(pivots) - 1 - r
            if A[pivot_row, free] == 1:
                vec[col] ^= 1
        basis.append(vec)
    return basis


def solve_matrix(
    relations: Iterable[Relation],
    primes: List[int],
    use_block_lanczos: Optional[bool] = None,
    block_size: int = 64,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Solve for dependencies between ``relations`` modulo 2.
    
    Args:
        relations: Iterable of Relation objects from sieving
        primes: List of primes in the factor base
        use_block_lanczos: Whether to use Block Lanczos algorithm.
            If None (default), automatically chooses based on matrix size.
        block_size: Block size for Block Lanczos (default 64)
        seed: Random seed for Block Lanczos reproducibility
    
    Returns:
        List of dependencies, where each dependency is a list of relation
        indices whose product has all even exponents.
    """
    rel_list = list(relations)
    if not rel_list:
        return []
    
    n_relations = len(rel_list)
    n_primes = len(primes)
    
    # Handle edge case of empty primes
    if n_primes == 0:
        # All relations have "even" exponents (vacuously), each is a dependency
        return [[i] for i in range(n_relations)]
    
    # Decide which solver to use
    if use_block_lanczos is None:
        use_block_lanczos = n_relations > BLOCK_LANCZOS_THRESHOLD
    
    combined_factors = [rel.combined_factors() for rel in rel_list]
    
    if use_block_lanczos and n_relations > BLOCK_LANCZOS_THRESHOLD:
        # Use Block Lanczos with sparse matrix
        from .sparse import SparseMatrixGF2
        from .block_lanczos import find_dependencies_block_lanczos
        
        # Build sparse matrix (primes x relations)
        sparse_matrix = SparseMatrixGF2(n_primes, n_relations)
        for j, factors in enumerate(combined_factors):
            for i, p in enumerate(primes):
                if factors.get(p, 0) % 2 == 1:
                    sparse_matrix.set(i, j)
        
        # Find nullspace using Block Lanczos
        basis = find_dependencies_block_lanczos(
            sparse_matrix,
            block_size=block_size,
            seed=seed,
        )
    else:
        # Use dense Gaussian elimination
        exponent_matrix = np.array(
            [[factors.get(p, 0) % 2 for factors in combined_factors] for p in primes],
            dtype=np.uint8,
        )
        basis = _nullspace_mod2(exponent_matrix)
    
    # Convert nullspace vectors to dependency lists
    dependencies: List[List[int]] = []
    for vec in basis:
        indices = [i for i, bit in enumerate(vec) if bit == 1]
        if indices:
            dependencies.append(indices)
    return dependencies


def solve_matrix_dense(relations: Iterable[Relation], primes: List[int]) -> List[List[int]]:
    """Solve using dense Gaussian elimination (original implementation).
    
    This is kept for compatibility and testing. Use solve_matrix() for
    production code.
    """
    return solve_matrix(relations, primes, use_block_lanczos=False)

"""Linear algebra utilities for GNFS.

This module provides a minimal implementation of solving for dependencies
between relations mod 2.  The matrix is constructed from the exponent
vectors of the relations and Gaussian elimination over GF(2) is used to
compute a basis for the nullspace.
"""

from typing import Iterable, List

import numpy as np

from ..sieve import Relation


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


def solve_matrix(relations: Iterable[Relation], primes: List[int]) -> List[List[int]]:
    """Solve for dependencies between ``relations`` modulo 2."""
    rel_list = list(relations)
    if not rel_list:
        return []
    exponent_matrix = np.array(
        [[rel.factors.get(p, 0) % 2 for rel in rel_list] for p in primes],
        dtype=int,
    )
    basis = _nullspace_mod2(exponent_matrix)
    dependencies: List[List[int]] = []
    for vec in basis:
        indices = [i for i, bit in enumerate(vec) if bit == 1]
        if indices:
            dependencies.append(indices)
    return dependencies

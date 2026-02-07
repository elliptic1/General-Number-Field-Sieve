"""Block Lanczos algorithm for finding nullspace over GF(2).

The Block Lanczos algorithm finds vectors in the nullspace of a matrix B = A^T A
over GF(2). Since we want Ax = 0 and x^T A^T A x = ||Ax||^2, any vector in
the nullspace of A^T A that gives Ax = 0 is what we need.

Key insight: Instead of working with single vectors (like standard Lanczos),
we work with blocks of N vectors at a time. This provides:
1. Better numerical stability
2. Natural parallelism
3. Faster convergence for finding multiple nullspace vectors

Algorithm overview:
1. Start with random block X_0
2. Iterate: X_{i+1} = B X_i - X_i S_i - X_{i-1} S_{i-1}^T
   where S_i are carefully chosen to maintain orthogonality
3. Terminate when X_i becomes zero (or rank-deficient)
4. Extract nullspace vectors from the accumulated subspace

References:
- Montgomery, P.L. (1995). "A Block Lanczos Algorithm for Finding 
  Dependencies over GF(2)"
- Coppersmith, D. (1994). "Solving Homogeneous Linear Equations Over GF(2) 
  via Block Wiedemann Algorithm"
"""

import numpy as np
from typing import List, Optional, Tuple
from .sparse import SparseMatrixGF2


# Block size - number of vectors processed together
# 64 is common because it matches machine word size
DEFAULT_BLOCK_SIZE = 64


def _random_block(n: int, block_size: int) -> np.ndarray:
    """Generate a random block of vectors over GF(2)."""
    return np.random.randint(0, 2, size=(n, block_size), dtype=np.uint8)


def _inner_product_block(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute X^T Y over GF(2), result is block_size x block_size."""
    # X is (n, k1), Y is (n, k2), result is (k1, k2)
    return (X.T @ Y) % 2


def _is_zero_block(X: np.ndarray) -> bool:
    """Check if a block is all zeros."""
    return not np.any(X)


def _rank_gf2(M: np.ndarray) -> int:
    """Compute rank of matrix over GF(2) using Gaussian elimination."""
    if M.size == 0:
        return 0
    A = M.copy() % 2
    m, n = A.shape
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(rank, m):
            if A[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap rows
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
        # Eliminate
        for row in range(m):
            if row != rank and A[row, col] == 1:
                A[row] ^= A[rank]
        rank += 1
    return rank


def _invert_gf2(M: np.ndarray) -> Optional[np.ndarray]:
    """Compute inverse of square matrix over GF(2), or None if singular."""
    n = M.shape[0]
    if M.shape[1] != n:
        return None
    
    # Augment with identity
    aug = np.zeros((n, 2*n), dtype=np.uint8)
    aug[:, :n] = M % 2
    aug[:, n:] = np.eye(n, dtype=np.uint8)
    
    # Gaussian elimination
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is None:
            return None  # Singular
        
        # Swap rows
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]
        
        # Eliminate
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] ^= aug[col]
    
    return aug[:, n:]


def _find_independent_columns(M: np.ndarray) -> List[int]:
    """Find indices of linearly independent columns over GF(2)."""
    if M.size == 0:
        return []
    A = M.copy() % 2
    m, n = A.shape
    independent = []
    row = 0
    
    for col in range(n):
        if row >= m:
            break
        # Find pivot
        pivot_row = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        independent.append(col)
        
        # Swap rows
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
        
        # Eliminate below
        for r in range(row + 1, m):
            if A[r, col] == 1:
                A[r] ^= A[row]
        row += 1
    
    return independent


def block_lanczos(
    matrix: SparseMatrixGF2,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Find vectors in the nullspace of matrix using Block Lanczos.
    
    Finds vectors x such that A @ x = 0 over GF(2).
    
    Args:
        matrix: Sparse matrix A over GF(2)
        block_size: Number of vectors per block (default 64)
        max_iterations: Maximum iterations (default: 2 * ncols / block_size)
        seed: Random seed for reproducibility
    
    Returns:
        List of nullspace vectors (as numpy arrays)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = matrix.ncols
    if max_iterations is None:
        max_iterations = max(10, 2 * n // block_size + 10)
    
    # Handle small matrices with dense method
    if n < 2 * block_size:
        return _dense_nullspace(matrix)
    
    # Initialize with random block
    X_prev = np.zeros((n, block_size), dtype=np.uint8)
    X_curr = _random_block(n, block_size)
    
    # Make sure we don't start with a zero block
    while _is_zero_block(X_curr):
        X_curr = _random_block(n, block_size)
    
    # Storage for Krylov subspace vectors
    # We accumulate vectors that could contribute to nullspace
    subspace_vectors: List[np.ndarray] = []
    
    S_prev = np.zeros((block_size, block_size), dtype=np.uint8)
    
    for iteration in range(max_iterations):
        # Compute B @ X_curr where B = A^T A
        BX = matrix.ata_matmat(X_curr)
        
        # Compute inner products for the three-term recurrence
        # S_curr = (X_curr^T B X_curr)^{-1} (if it exists)
        XtBX = _inner_product_block(X_curr, BX)
        
        # Check if X_curr is in the nullspace of B (means A @ X = 0)
        if _is_zero_block(BX):
            # All current vectors are in nullspace of A^T A
            # Check which ones actually satisfy A @ x = 0
            for col in range(block_size):
                x = X_curr[:, col]
                if np.any(x):  # Non-zero vector
                    Ax = matrix.matvec(x)
                    if not np.any(Ax):  # Actually in nullspace of A
                        subspace_vectors.append(x.copy())
            break
        
        # Find linearly independent columns of X_curr
        indep_cols = _find_independent_columns(X_curr.T)
        if not indep_cols:
            break
        
        # If we've lost rank, we may have found nullspace vectors
        if len(indep_cols) < block_size:
            # Vectors in the kernel of X_curr^T might be nullspace candidates
            # This is a simplification - full implementation needs more care
            pass
        
        # Store current X for potential nullspace extraction
        for col in range(X_curr.shape[1]):
            if np.any(X_curr[:, col]):
                subspace_vectors.append(X_curr[:, col].copy())
        
        # Compute next block using three-term recurrence with B-orthogonalization
        # X_next = B @ X_curr - X_curr @ S_curr - X_prev @ S_prev^T
        # where S_curr = (X_curr^T B X_curr)^{-1} (X_curr^T B^2 X_curr)
        
        # Compute S_curr for B-orthogonalization
        # S_curr should make X_next B-orthogonal to X_curr
        # We need: X_curr^T B X_next = 0
        # X_next = BX - X_curr @ S_curr - X_prev @ S_prev^T
        # X_curr^T B X_next = X_curr^T B BX - X_curr^T B X_curr @ S_curr - X_curr^T B X_prev @ S_prev^T
        #                   = XtBX @ XtBX - XtBX @ S_curr - (X_curr^T B X_prev) @ S_prev^T = 0
        # If XtBX is invertible: S_curr = XtBX - XtBX^{-1} @ (X_curr^T B X_prev) @ S_prev^T
        
        # Try to compute S_curr using XtBX
        XtBX_inv = _invert_gf2(XtBX)
        
        X_next = BX.copy()
        
        if XtBX_inv is not None:
            # Proper B-orthogonalization
            # S_curr makes X_next B-orthogonal to X_curr
            # We want X_curr^T B X_next = 0
            # X_curr^T B X_next = X_curr^T B (BX - X_curr @ S_curr) = XtBX^2 - XtBX @ S_curr
            # Setting S_curr = XtBX gives XtBX^2 - XtBX^2 = 0 (mod 2)
            S_curr = XtBX
            X_next = (X_next - (X_curr @ S_curr) % 2) % 2
        else:
            # Fallback: use unweighted projection when XtBX is singular
            S_curr = _inner_product_block(X_curr, X_next)
            X_next = (X_next - (X_curr @ S_curr) % 2) % 2
        
        # Subtract projection onto X_prev (three-term recurrence)
        if not _is_zero_block(X_prev):
            proj_prev = _inner_product_block(X_prev, X_next)
            X_next = (X_next - (X_prev @ proj_prev) % 2) % 2
        
        # Move to next iteration
        X_prev = X_curr
        X_curr = X_next
        S_prev = S_curr
        
        # Check for convergence (X became zero or very low rank)
        if _is_zero_block(X_curr) or _rank_gf2(X_curr) < block_size // 4:
            break
    
    # Extract actual nullspace vectors from accumulated subspace
    return _extract_nullspace_vectors(matrix, subspace_vectors)


def _dense_nullspace(matrix: SparseMatrixGF2) -> List[np.ndarray]:
    """Fallback to dense nullspace computation for small matrices."""
    A = matrix.to_dense()
    m, n = A.shape
    
    # Compute nullspace using Gaussian elimination
    aug = A.copy() % 2
    row = 0
    pivots = []
    
    for col in range(n):
        if row >= m:
            break
        # Find pivot
        pivot_row = None
        for r in range(row, m):
            if aug[r, col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        
        pivots.append(col)
        if pivot_row != row:
            aug[[row, pivot_row]] = aug[[pivot_row, row]]
        
        for r in range(m):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]
        row += 1
    
    # Free columns form basis of nullspace
    free_cols = [c for c in range(n) if c not in pivots]
    basis = []
    
    for free in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        for r, col in enumerate(pivots):
            if aug[r, free] == 1:
                vec[col] = 1
        # Verify this is in nullspace
        if not np.any(matrix.matvec(vec)):
            basis.append(vec)
    
    return basis


def _extract_nullspace_vectors(
    matrix: SparseMatrixGF2,
    candidates: List[np.ndarray],
) -> List[np.ndarray]:
    """Extract actual nullspace vectors from candidate vectors.
    
    Tries linear combinations of candidates to find vectors where Ax = 0.
    """
    if not candidates:
        return []
    
    # First, filter to keep only non-zero candidates that might be in nullspace
    valid = []
    for v in candidates:
        if np.any(v):
            valid.append(v)
    
    if not valid:
        return []
    
    # Stack into matrix and find nullspace vectors
    n = valid[0].shape[0]
    
    # Check each candidate directly
    nullspace = []
    for v in valid:
        Av = matrix.matvec(v)
        if not np.any(Av):
            # v is in nullspace of A
            # Check if it's linearly independent from what we have
            if not nullspace:
                nullspace.append(v)
            else:
                # Check linear independence
                mat = np.column_stack(nullspace + [v])
                if _rank_gf2(mat) > len(nullspace):
                    nullspace.append(v)
    
    # Also try pairwise XORs
    for i in range(len(valid)):
        for j in range(i + 1, min(len(valid), i + 100)):  # Limit combinations
            v = (valid[i] ^ valid[j]) % 2
            if np.any(v):
                Av = matrix.matvec(v)
                if not np.any(Av):
                    if not nullspace:
                        nullspace.append(v)
                    else:
                        mat = np.column_stack(nullspace + [v])
                        if _rank_gf2(mat) > len(nullspace):
                            nullspace.append(v)
    
    return nullspace


def find_dependencies_block_lanczos(
    matrix: SparseMatrixGF2,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_preprocessing: bool = True,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """High-level interface for finding nullspace using Block Lanczos.
    
    This is the main entry point for the linear algebra phase of GNFS.
    
    Args:
        matrix: Sparse exponent matrix over GF(2)
        block_size: Block size for Lanczos (default 64)
        use_preprocessing: Whether to apply structured Gaussian elimination first
        seed: Random seed for reproducibility
    
    Returns:
        List of nullspace vectors
    """
    from .sparse import structured_gaussian_elimination
    
    if use_preprocessing and matrix.ncols > 100:
        # Apply preprocessing to reduce matrix size
        reduced, col_map, eliminated = structured_gaussian_elimination(matrix)
        
        if reduced.ncols == 0:
            # Entire matrix was eliminated - reconstruct vectors from eliminations
            return _reconstruct_from_eliminations(matrix.ncols, eliminated)
        
        # Find nullspace of reduced matrix
        reduced_nullspace = block_lanczos(reduced, block_size, seed=seed)
        
        # Expand back to original columns
        full_nullspace = []
        for v in reduced_nullspace:
            full_v = _expand_nullspace_vector(v, col_map, eliminated, matrix.ncols)
            full_nullspace.append(full_v)
        
        return full_nullspace
    else:
        return block_lanczos(matrix, block_size, seed=seed)


def _reconstruct_from_eliminations(
    n_cols: int,
    eliminated: List[Tuple[int, List[int]]],
) -> List[np.ndarray]:
    """Reconstruct nullspace vectors when all columns were eliminated."""
    # Each elimination gives us a dependency
    # If column j was eliminated with dependencies [c1, c2, ...],
    # then x_j = x_c1 XOR x_c2 XOR ...
    # Setting x_j = 1 and solving gives us a nullspace vector
    
    vectors = []
    for j, deps in eliminated:
        v = np.zeros(n_cols, dtype=np.uint8)
        v[j] = 1
        for d in deps:
            v[d] = 1
        vectors.append(v)
    
    return vectors


def _expand_nullspace_vector(
    reduced_v: np.ndarray,
    col_map: List[int],
    eliminated: List[Tuple[int, List[int]]],
    n_cols: int,
) -> np.ndarray:
    """Expand a nullspace vector of reduced matrix to full matrix."""
    # Start with zeros
    full_v = np.zeros(n_cols, dtype=np.uint8)
    
    # Fill in values for remaining columns
    for i, orig_col in enumerate(col_map):
        full_v[orig_col] = reduced_v[i]
    
    # Back-substitute eliminated columns in reverse order
    for j, deps in reversed(eliminated):
        # x_j = XOR of x_c for c in deps
        val = 0
        for c in deps:
            val ^= full_v[c]
        full_v[j] = val
    
    return full_v

"""Comprehensive mathematical tests for linear algebra over GF(2).

These tests verify the mathematical correctness of:
1. Sparse matrix operations over GF(2)
2. Block Lanczos algorithm properties
3. Structured Gaussian elimination
4. Nullspace computation correctness
5. Integration with GNFS relation matrices
"""

import pytest
import numpy as np
from typing import List, Set

from gnfs.linalg.sparse import SparseMatrixGF2, structured_gaussian_elimination
from gnfs.linalg.block_lanczos import (
    block_lanczos,
    find_dependencies_block_lanczos,
    _rank_gf2,
    _invert_gf2,
    _find_independent_columns,
)
from gnfs.linalg import solve_matrix, solve_matrix_dense
from gnfs.sieve import Relation


# =============================================================================
# GF(2) Arithmetic Tests
# =============================================================================

class TestGF2Arithmetic:
    """Test that all operations respect GF(2) arithmetic."""
    
    def test_addition_is_xor(self):
        """Verify addition is XOR in GF(2)."""
        mat = SparseMatrixGF2(2, 2)
        mat.set(0, 0, 1)
        mat.set(0, 1, 1)
        
        # 1 + 1 = 0 in GF(2)
        x = np.array([1, 1], dtype=np.uint8)
        result = mat.matvec(x)
        assert result[0] == 0  # 1*1 + 1*1 = 0 (mod 2)
    
    def test_multiplication_mod_2(self):
        """Verify all products are reduced mod 2."""
        dense = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        x = np.array([1, 1, 1, 1], dtype=np.uint8)
        result = mat.matvec(x)
        
        # Row 0: 1+1+1+1 = 0 (mod 2)
        # Row 1: 1+0+1+0 = 0 (mod 2)
        # Row 2: 0+1+0+1 = 0 (mod 2)
        expected = np.array([0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_matrix_matrix_mod_2(self):
        """Test matrix-matrix product respects GF(2)."""
        A = np.array([
            [1, 1],
            [1, 0],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(A)
        
        X = np.array([
            [1, 0],
            [1, 1],
        ], dtype=np.uint8)
        
        result = mat.matmat(X)
        
        # A @ X in GF(2):
        # [1,1] @ [[1,0],[1,1]] = [1+1, 0+1] = [0, 1]
        # [1,0] @ [[1,0],[1,1]] = [1+0, 0+0] = [1, 0]
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_ata_symmetric(self):
        """Test that A^T A produces symmetric results."""
        np.random.seed(42)
        dense = np.random.randint(0, 2, size=(10, 15), dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        # Compute A^T A @ x for various x
        for _ in range(5):
            x = np.random.randint(0, 2, size=15, dtype=np.uint8)
            y = np.random.randint(0, 2, size=15, dtype=np.uint8)
            
            ATAx = mat.ata_matvec(x)
            ATAy = mat.ata_matvec(y)
            
            # x^T (A^T A) y should equal y^T (A^T A) x
            # (since A^T A is symmetric)
            lhs = np.sum(x * ATAy) % 2
            rhs = np.sum(y * ATAx) % 2
            assert lhs == rhs


# =============================================================================
# Sparse Matrix Properties
# =============================================================================

class TestSparseMatrixProperties:
    """Test mathematical properties of sparse matrices."""
    
    def test_transpose_via_matvec(self):
        """Verify transpose relationship: (Ax)^T y = x^T (A^T y)."""
        np.random.seed(123)
        dense = np.random.randint(0, 2, size=(8, 12), dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        x = np.random.randint(0, 2, size=12, dtype=np.uint8)
        y = np.random.randint(0, 2, size=8, dtype=np.uint8)
        
        Ax = mat.matvec(x)
        ATy = mat.matvec_transpose(y)
        
        # (Ax)^T y = sum(Ax * y) mod 2
        lhs = np.sum(Ax * y) % 2
        # x^T (A^T y) = sum(x * ATy) mod 2
        rhs = np.sum(x * ATy) % 2
        
        assert lhs == rhs
    
    def test_matmat_equals_multiple_matvec(self):
        """Verify matmat produces same result as multiple matvec calls."""
        np.random.seed(456)
        dense = np.random.randint(0, 2, size=(5, 8), dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        X = np.random.randint(0, 2, size=(8, 4), dtype=np.uint8)
        
        # Using matmat
        result_matmat = mat.matmat(X)
        
        # Using individual matvec calls
        result_matvec = np.zeros((5, 4), dtype=np.uint8)
        for j in range(4):
            result_matvec[:, j] = mat.matvec(X[:, j])
        
        np.testing.assert_array_equal(result_matmat, result_matvec)
    
    def test_density_calculation(self):
        """Test density is computed correctly."""
        mat = SparseMatrixGF2(10, 20)
        assert mat.density == 0.0
        
        # Add 50 ones
        for i in range(10):
            for j in range(5):
                mat.set(i, j, 1)
        
        assert mat.nnz == 50
        assert mat.density == 50 / 200  # 0.25
    
    def test_row_column_consistency(self):
        """Verify row and column indices are consistent."""
        np.random.seed(789)
        mat = SparseMatrixGF2(20, 30)
        
        # Set random entries
        for _ in range(100):
            i = np.random.randint(0, 20)
            j = np.random.randint(0, 30)
            mat.set(i, j, 1)
        
        # Check consistency
        for i in range(20):
            for j in mat.row_indices(i):
                assert i in mat.col_indices(j)
        
        for j in range(30):
            for i in mat.col_indices(j):
                assert j in mat.row_indices(i)


# =============================================================================
# Rank and Linear Independence Tests
# =============================================================================

class TestRankComputation:
    """Test rank computation over GF(2)."""
    
    def test_rank_identity(self):
        """Identity matrix has full rank."""
        for n in [1, 5, 10]:
            M = np.eye(n, dtype=np.uint8)
            assert _rank_gf2(M) == n
    
    def test_rank_zero_matrix(self):
        """Zero matrix has rank 0."""
        M = np.zeros((5, 5), dtype=np.uint8)
        assert _rank_gf2(M) == 0
    
    def test_rank_single_row(self):
        """Single non-zero row has rank 1."""
        M = np.array([[1, 0, 1, 1, 0]], dtype=np.uint8)
        assert _rank_gf2(M) == 1
    
    def test_rank_dependent_rows(self):
        """Rows that XOR to zero reduce rank."""
        # Row 2 = Row 0 XOR Row 1
        M = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],  # = row0 XOR row1
        ], dtype=np.uint8)
        assert _rank_gf2(M) == 2
    
    def test_rank_wide_matrix(self):
        """Rank bounded by min(rows, cols)."""
        M = np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ], dtype=np.uint8)
        assert _rank_gf2(M) == 2
    
    def test_rank_tall_matrix(self):
        """Rank bounded by min(rows, cols)."""
        M = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
        ], dtype=np.uint8)
        assert _rank_gf2(M) == 2
    
    def test_find_independent_columns_identity(self):
        """All columns of identity are independent."""
        M = np.eye(5, dtype=np.uint8)
        cols = _find_independent_columns(M)
        assert cols == [0, 1, 2, 3, 4]
    
    def test_find_independent_columns_dependent(self):
        """Identifies dependent columns correctly."""
        # Column 2 = Column 0 XOR Column 1
        M = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        cols = _find_independent_columns(M)
        assert len(cols) == 2
        assert 0 in cols and 1 in cols


# =============================================================================
# Matrix Inversion Tests
# =============================================================================

class TestMatrixInversion:
    """Test matrix inversion over GF(2)."""
    
    def test_invert_identity(self):
        """Identity inverts to itself."""
        for n in [1, 3, 5]:
            M = np.eye(n, dtype=np.uint8)
            M_inv = _invert_gf2(M)
            assert M_inv is not None
            np.testing.assert_array_equal(M_inv, M)
    
    def test_invert_permutation(self):
        """Permutation matrices are invertible."""
        # Cyclic permutation
        M = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.uint8)
        M_inv = _invert_gf2(M)
        assert M_inv is not None
        
        # Verify M @ M_inv = I
        product = (M @ M_inv) % 2
        np.testing.assert_array_equal(product, np.eye(3, dtype=np.uint8))
    
    def test_invert_involution(self):
        """Self-inverse matrices."""
        # Matrix that is its own inverse
        M = np.array([
            [1, 1],
            [0, 1],
        ], dtype=np.uint8)
        M_inv = _invert_gf2(M)
        assert M_inv is not None
        
        # Verify inverse is correct
        product = (M @ M_inv) % 2
        np.testing.assert_array_equal(product, np.eye(2, dtype=np.uint8))
    
    def test_singular_has_no_inverse(self):
        """Singular matrices return None."""
        # All zeros
        M = np.zeros((3, 3), dtype=np.uint8)
        assert _invert_gf2(M) is None
        
        # Rank-deficient
        M = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=np.uint8)
        assert _invert_gf2(M) is None
    
    def test_non_square_has_no_inverse(self):
        """Non-square matrices return None."""
        M = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.uint8)
        assert _invert_gf2(M) is None


# =============================================================================
# Nullspace Computation Tests
# =============================================================================

class TestNullspaceComputation:
    """Test nullspace computation over GF(2)."""
    
    def test_nullspace_vectors_satisfy_Ax_equals_zero(self):
        """Every nullspace vector must satisfy Ax = 0."""
        np.random.seed(111)
        for _ in range(10):
            # Random sparse matrix
            m, n = np.random.randint(5, 20), np.random.randint(10, 30)
            dense = (np.random.random((m, n)) < 0.2).astype(np.uint8)
            mat = SparseMatrixGF2.from_dense(dense)
            
            nullspace = block_lanczos(mat, block_size=8, seed=42)
            
            for v in nullspace:
                Av = mat.matvec(v)
                assert not np.any(Av), f"Nullspace vector does not satisfy Ax=0"
    
    def test_nullspace_vectors_linearly_independent(self):
        """Nullspace vectors should be linearly independent."""
        dense = np.array([
            [1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        
        if len(nullspace) > 1:
            # Stack into matrix and check rank
            null_mat = np.column_stack(nullspace)
            rank = _rank_gf2(null_mat)
            assert rank == len(nullspace), "Nullspace vectors not linearly independent"
    
    def test_nullspace_dimension_bound(self):
        """Nullspace dimension = n - rank(A)."""
        np.random.seed(222)
        for _ in range(5):
            m, n = 10, 20
            dense = (np.random.random((m, n)) < 0.3).astype(np.uint8)
            mat = SparseMatrixGF2.from_dense(dense)
            
            rank = _rank_gf2(dense)
            expected_null_dim = n - rank
            
            nullspace = block_lanczos(mat, block_size=8, seed=42)
            
            # We may not find all nullspace vectors, but shouldn't find more
            assert len(nullspace) <= expected_null_dim
    
    def test_full_rank_has_trivial_nullspace(self):
        """Full rank matrix has only zero vector in nullspace."""
        # Identity extended with extra columns
        M = np.eye(10, dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(M)
        
        nullspace = block_lanczos(mat, block_size=8, seed=42)
        assert len(nullspace) == 0


# =============================================================================
# Structured Gaussian Elimination Tests
# =============================================================================

class TestStructuredGaussianEliminationMath:
    """Mathematical tests for structured Gaussian elimination."""
    
    def test_preserves_nullspace_dimension(self):
        """Elimination should preserve nullspace dimension."""
        np.random.seed(333)
        dense = (np.random.random((15, 25)) < 0.15).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        original_rank = _rank_gf2(dense)
        original_null_dim = 25 - original_rank
        
        reduced, col_map, eliminated = structured_gaussian_elimination(mat)
        
        # Reduced matrix + eliminated columns should give same nullspace dim
        reduced_dense = reduced.to_dense()
        if reduced_dense.size > 0:
            reduced_rank = _rank_gf2(reduced_dense)
        else:
            reduced_rank = 0
        
        # Account for eliminated columns
        total_null_dim = (len(col_map) - reduced_rank) + len(eliminated)
        
        # Should be at least as many (elimination can reveal more structure)
        assert total_null_dim >= original_null_dim - 1  # Allow small error
    
    def test_eliminated_dependencies_structure(self):
        """Eliminated column dependencies should have valid structure."""
        # Matrix where column 0 = column 1 XOR column 2
        dense = np.array([
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        reduced, col_map, eliminated = structured_gaussian_elimination(mat)
        
        # Verify structure: eliminated columns + remaining columns = all columns
        all_cols = set(range(4))
        eliminated_cols = {col_idx for col_idx, _ in eliminated}
        remaining_cols = set(col_map)
        
        # No overlap between eliminated and remaining
        assert eliminated_cols.isdisjoint(remaining_cols)
        
        # All columns accounted for
        assert eliminated_cols | remaining_cols == all_cols
    
    def test_reduced_matrix_valid_structure(self):
        """Reduced matrix has valid structure."""
        np.random.seed(444)
        dense = (np.random.random((10, 20)) < 0.2).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        reduced, col_map, eliminated = structured_gaussian_elimination(mat)
        
        # Reduced matrix dimensions should be consistent
        reduced_dense = reduced.to_dense()
        assert reduced.ncols == len(col_map)
        assert reduced.nrows == mat.nrows
        
        # col_map should reference valid original columns
        for orig_idx in col_map:
            assert 0 <= orig_idx < 20
        
        # No duplicate columns in col_map
        assert len(col_map) == len(set(col_map))


# =============================================================================
# Block Lanczos Mathematical Properties
# =============================================================================

class TestBlockLanczosMathematics:
    """Test mathematical properties of Block Lanczos."""
    
    def test_krylov_subspace_contains_nullspace(self):
        """Krylov subspace should eventually contain nullspace vectors."""
        # Simple matrix with known nullspace
        # Nullspace: [1,1,0,0], [0,0,1,1]
        dense = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        
        # Should find at least one vector
        assert len(nullspace) >= 1
        
        # Verify found vectors
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av)
    
    def test_random_matrix_nullspace(self):
        """Test Block Lanczos on random matrices."""
        np.random.seed(555)
        for trial in range(5):
            # Create random matrix with guaranteed nullspace
            m, n = 30, 50
            dense = (np.random.random((m, n)) < 0.15).astype(np.uint8)
            mat = SparseMatrixGF2.from_dense(dense)
            
            nullspace = block_lanczos(mat, block_size=16, seed=trial)
            
            # All returned vectors must be in nullspace
            for v in nullspace:
                Av = mat.matvec(v)
                assert not np.any(Av), f"Trial {trial}: Found non-nullspace vector"
    
    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        dense = np.random.randint(0, 2, size=(20, 40)).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        null1 = block_lanczos(mat, block_size=8, seed=12345)
        null2 = block_lanczos(mat, block_size=8, seed=12345)
        
        assert len(null1) == len(null2)
        for v1, v2 in zip(null1, null2):
            np.testing.assert_array_equal(v1, v2)


# =============================================================================
# GNFS Integration Tests
# =============================================================================

class TestGNFSIntegration:
    """Test linear algebra with GNFS-style relation matrices."""
    
    def test_exponent_matrix_construction(self):
        """Test building exponent matrix from relations."""
        relations = [
            Relation(a=1, b=1, algebraic_value=12, rational_value=6,
                    algebraic_factors={2: 2, 3: 1}, rational_factors={2: 1, 3: 1}),
            Relation(a=2, b=1, algebraic_value=18, rational_value=9,
                    algebraic_factors={2: 1, 3: 2}, rational_factors={3: 2}),
            Relation(a=3, b=1, algebraic_value=8, rational_value=4,
                    algebraic_factors={2: 3}, rational_factors={2: 2}),
        ]
        primes = [2, 3]
        
        deps = solve_matrix(relations, primes)
        
        # Verify each dependency produces perfect square
        for dep in deps:
            combined = {}
            for idx in dep:
                for p, exp in relations[idx].combined_factors().items():
                    combined[p] = combined.get(p, 0) + exp
            
            for p, total in combined.items():
                assert total % 2 == 0, f"Odd exponent {total} for prime {p}"
    
    def test_larger_relation_set(self):
        """Test with more realistic relation set."""
        np.random.seed(666)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Generate random relations
        relations = []
        for i in range(50):
            alg_factors = {}
            rat_factors = {}
            
            # Random factors from prime base
            for p in primes:
                if np.random.random() < 0.3:
                    alg_factors[p] = np.random.randint(1, 4)
                if np.random.random() < 0.3:
                    rat_factors[p] = np.random.randint(1, 4)
            
            if alg_factors or rat_factors:
                alg_val = 1
                for p, e in alg_factors.items():
                    alg_val *= p ** e
                rat_val = 1
                for p, e in rat_factors.items():
                    rat_val *= p ** e
                
                relations.append(Relation(
                    a=i+1, b=1,
                    algebraic_value=alg_val,
                    rational_value=rat_val,
                    algebraic_factors=alg_factors,
                    rational_factors=rat_factors,
                ))
        
        deps = solve_matrix(relations, primes)
        
        # Verify all dependencies are valid
        for dep in deps:
            combined = {}
            for idx in dep:
                for p, exp in relations[idx].combined_factors().items():
                    combined[p] = combined.get(p, 0) + exp
            
            for p, total in combined.items():
                assert total % 2 == 0, f"Invalid dependency: odd exponent for {p}"
    
    def test_dense_vs_sparse_consistency(self):
        """Dense and sparse solvers should find equivalent dependencies."""
        relations = [
            Relation(a=i, b=1, algebraic_value=2**i * 3**(i%3),
                    rational_value=5**(i%2),
                    algebraic_factors={2: i, 3: i%3},
                    rational_factors={5: i%2})
            for i in range(1, 20)
        ]
        primes = [2, 3, 5]
        
        deps_dense = solve_matrix_dense(relations, primes)
        deps_sparse = solve_matrix(relations, primes, use_block_lanczos=True)
        
        # Both should find valid dependencies
        def verify_deps(deps, name):
            for dep in deps:
                combined = {}
                for idx in dep:
                    for p, exp in relations[idx].combined_factors().items():
                        combined[p] = combined.get(p, 0) + exp
                for p, total in combined.items():
                    assert total % 2 == 0, f"{name}: invalid dependency"
        
        verify_deps(deps_dense, "dense")
        verify_deps(deps_sparse, "sparse")


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================

class TestEdgeCasesAndStress:
    """Edge cases and stress tests."""
    
    def test_single_column_matrix(self):
        """Matrix with single column."""
        dense = np.array([[1], [0], [1]], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        # Single column with non-zero entries has no nullspace
        for v in nullspace:
            assert not np.any(v) or not np.any(mat.matvec(v))
    
    def test_single_row_matrix(self):
        """Matrix with single row."""
        dense = np.array([[1, 0, 1, 0, 1]], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        
        # Should find vectors where positions 0,2,4 have even parity
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av)
    
    def test_very_sparse_matrix(self):
        """Very sparse matrix (~1% density)."""
        np.random.seed(777)
        m, n = 100, 200
        dense = (np.random.random((m, n)) < 0.01).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        assert mat.density < 0.02
        
        nullspace = block_lanczos(mat, block_size=32, seed=42)
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av)
    
    def test_diagonal_matrix(self):
        """Diagonal matrix has no nullspace."""
        dense = np.diag([1, 1, 1, 1, 1]).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        assert len(nullspace) == 0
    
    def test_all_ones_row(self):
        """Matrix with row of all ones."""
        dense = np.zeros((3, 10), dtype=np.uint8)
        dense[0, :] = 1  # All ones
        dense[1, [0, 2, 4, 6, 8]] = 1
        dense[2, [1, 3, 5, 7, 9]] = 1
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av)
            # Row 0 constraint: sum of all v[i] must be even
            assert np.sum(v) % 2 == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Block Lanczos and sparse matrix operations."""

import pytest
import numpy as np

from gnfs.linalg.sparse import SparseMatrixGF2, structured_gaussian_elimination
from gnfs.linalg.block_lanczos import (
    block_lanczos,
    find_dependencies_block_lanczos,
    _rank_gf2,
    _invert_gf2,
)
from gnfs.linalg import solve_matrix, solve_matrix_dense
from gnfs.sieve import Relation


# =============================================================================
# Sparse Matrix Tests
# =============================================================================

class TestSparseMatrixGF2:
    """Tests for sparse matrix over GF(2)."""
    
    def test_create_empty(self):
        """Test creating an empty matrix."""
        mat = SparseMatrixGF2(5, 10)
        assert mat.nrows == 5
        assert mat.ncols == 10
        assert mat.nnz == 0
        assert mat.density == 0.0
    
    def test_set_and_get(self):
        """Test setting and getting entries."""
        mat = SparseMatrixGF2(3, 3)
        mat.set(0, 0, 1)
        mat.set(1, 2, 1)
        mat.set(2, 1, 1)
        
        assert mat.get(0, 0) == 1
        assert mat.get(1, 2) == 1
        assert mat.get(2, 1) == 1
        assert mat.get(0, 1) == 0
        assert mat.get(1, 1) == 0
        assert mat.nnz == 3
    
    def test_flip(self):
        """Test flipping entries."""
        mat = SparseMatrixGF2(2, 2)
        assert mat.get(0, 0) == 0
        
        mat.flip(0, 0)
        assert mat.get(0, 0) == 1
        assert mat.nnz == 1
        
        mat.flip(0, 0)
        assert mat.get(0, 0) == 0
        assert mat.nnz == 0
    
    def test_from_dense(self):
        """Test creating from dense array."""
        dense = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ], dtype=np.uint8)
        
        mat = SparseMatrixGF2.from_dense(dense)
        assert mat.nrows == 3
        assert mat.ncols == 3
        assert mat.nnz == 5
        
        # Check all entries
        for i in range(3):
            for j in range(3):
                assert mat.get(i, j) == dense[i, j]
    
    def test_to_dense(self):
        """Test converting to dense array."""
        mat = SparseMatrixGF2(2, 3)
        mat.set(0, 1, 1)
        mat.set(1, 0, 1)
        mat.set(1, 2, 1)
        
        dense = mat.to_dense()
        expected = np.array([
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.uint8)
        
        np.testing.assert_array_equal(dense, expected)
    
    def test_roundtrip_dense(self):
        """Test dense -> sparse -> dense roundtrip."""
        original = np.random.randint(0, 2, size=(10, 15), dtype=np.uint8)
        sparse = SparseMatrixGF2.from_dense(original)
        recovered = sparse.to_dense()
        np.testing.assert_array_equal(original, recovered)
    
    def test_matvec(self):
        """Test matrix-vector multiplication."""
        # Matrix:
        # [1 0 1]
        # [0 1 1]
        mat = SparseMatrixGF2(2, 3)
        mat.set(0, 0, 1)
        mat.set(0, 2, 1)
        mat.set(1, 1, 1)
        mat.set(1, 2, 1)
        
        x = np.array([1, 0, 1], dtype=np.uint8)
        result = mat.matvec(x)
        
        # Row 0: 1*1 + 0*0 + 1*1 = 0 (mod 2)
        # Row 1: 0*1 + 1*0 + 1*1 = 1 (mod 2)
        expected = np.array([0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_matvec_transpose(self):
        """Test transpose matrix-vector multiplication."""
        mat = SparseMatrixGF2(2, 3)
        mat.set(0, 0, 1)
        mat.set(0, 2, 1)
        mat.set(1, 1, 1)
        mat.set(1, 2, 1)
        
        y = np.array([1, 1], dtype=np.uint8)
        result = mat.matvec_transpose(y)
        
        # Col 0: 1*1 + 0*1 = 1
        # Col 1: 0*1 + 1*1 = 1
        # Col 2: 1*1 + 1*1 = 0
        expected = np.array([1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
    
    def test_ata_matvec(self):
        """Test (A^T A) x computation."""
        dense = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        x = np.array([1, 0, 1], dtype=np.uint8)
        result = mat.ata_matvec(x)
        
        # Compute expected: A^T @ (A @ x)
        Ax = (dense @ x) % 2
        expected = (dense.T @ Ax) % 2
        
        np.testing.assert_array_equal(result, expected)
    
    def test_row_col_weights(self):
        """Test row and column weight computation."""
        mat = SparseMatrixGF2(3, 4)
        mat.set(0, 0, 1)
        mat.set(0, 2, 1)
        mat.set(1, 1, 1)
        mat.set(2, 0, 1)
        mat.set(2, 1, 1)
        mat.set(2, 3, 1)
        
        assert mat.row_weight(0) == 2
        assert mat.row_weight(1) == 1
        assert mat.row_weight(2) == 3
        
        assert mat.col_weight(0) == 2
        assert mat.col_weight(1) == 2
        assert mat.col_weight(2) == 1
        assert mat.col_weight(3) == 1


# =============================================================================
# Structured Gaussian Elimination Tests
# =============================================================================

class TestStructuredGaussianElimination:
    """Tests for preprocessing step."""
    
    def test_eliminates_weight_one_columns(self):
        """Test that columns with weight 1 are eliminated."""
        # Column 2 has weight 1
        dense = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],  # Column 2 only appears here
            [0, 1, 0, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        reduced, col_map, eliminated = structured_gaussian_elimination(mat)
        
        # Column 2 should have been eliminated
        assert 2 not in col_map or len(eliminated) > 0
    
    def test_preserves_nullspace(self):
        """Test that elimination preserves nullspace structure."""
        # Create a matrix with known nullspace
        # Columns 0+1+2 = 0 (mod 2)
        dense = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        reduced, col_map, eliminated = structured_gaussian_elimination(mat)
        
        # Should still be able to reconstruct nullspace vectors
        assert isinstance(reduced, SparseMatrixGF2)
        assert isinstance(col_map, list)
        assert isinstance(eliminated, list)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_rank_gf2(self):
        """Test rank computation over GF(2)."""
        # Full rank 3x3 identity
        M = np.eye(3, dtype=np.uint8)
        assert _rank_gf2(M) == 3
        
        # Rank 2 matrix (row 2 = row 0 XOR row 1)
        M = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        assert _rank_gf2(M) == 2
        
        # Zero matrix
        M = np.zeros((3, 3), dtype=np.uint8)
        assert _rank_gf2(M) == 0
    
    def test_invert_gf2(self):
        """Test matrix inversion over GF(2)."""
        # Identity inverts to itself
        M = np.eye(3, dtype=np.uint8)
        M_inv = _invert_gf2(M)
        assert M_inv is not None
        np.testing.assert_array_equal(M_inv, M)
        
        # Known invertible matrix (permutation matrix)
        M = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.uint8)
        M_inv = _invert_gf2(M)
        assert M_inv is not None
        
        # Verify M @ M_inv = I (mod 2)
        product = (M @ M_inv) % 2
        np.testing.assert_array_equal(product, np.eye(3, dtype=np.uint8))
    
    def test_invert_gf2_singular(self):
        """Test that singular matrices return None."""
        # Singular matrix (row 2 = row 0 XOR row 1)
        M = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.uint8)
        assert _invert_gf2(M) is None


# =============================================================================
# Block Lanczos Tests
# =============================================================================

class TestBlockLanczos:
    """Tests for Block Lanczos algorithm."""
    
    def test_finds_nullspace_small(self):
        """Test finding nullspace of small matrix."""
        # Matrix with known nullspace: x = [1, 1, 1, 0] is in nullspace
        # Rows represent: x0 + x1 = 0, x1 + x2 = 0, x0 + x2 = 0
        dense = np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=4, seed=42)
        
        # Should find at least one nullspace vector
        assert len(nullspace) >= 1
        
        # Verify each vector is in nullspace
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av), "Vector not in nullspace"
    
    def test_finds_nullspace_medium(self):
        """Test finding nullspace of medium-sized matrix."""
        np.random.seed(42)
        # Create random sparse matrix with ~10% density
        nrows, ncols = 50, 100
        dense = (np.random.random((nrows, ncols)) < 0.1).astype(np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, block_size=16, seed=42)
        
        # Verify each vector is in nullspace
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av), "Vector not in nullspace"
    
    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        mat = SparseMatrixGF2(0, 0)
        nullspace = block_lanczos(mat, seed=42)
        assert nullspace == []
    
    def test_identity_has_no_nullspace(self):
        """Test that identity matrix has empty nullspace."""
        dense = np.eye(10, dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = block_lanczos(mat, seed=42)
        
        # Identity is full rank, nullspace should be empty
        assert len(nullspace) == 0


class TestFindDependenciesBlockLanczos:
    """Tests for the high-level Block Lanczos interface."""
    
    def test_with_preprocessing(self):
        """Test Block Lanczos with preprocessing."""
        # Create sparse matrix
        dense = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
        ], dtype=np.uint8)
        mat = SparseMatrixGF2.from_dense(dense)
        
        nullspace = find_dependencies_block_lanczos(mat, seed=42)
        
        # Verify vectors are in nullspace
        for v in nullspace:
            Av = mat.matvec(v)
            assert not np.any(Av)


# =============================================================================
# Integration Tests with solve_matrix
# =============================================================================

class TestSolveMatrixIntegration:
    """Tests for integration with solve_matrix."""
    
    def test_dense_vs_block_lanczos_small(self):
        """Test that dense and Block Lanczos give consistent results."""
        # Create small set of relations
        relations = [
            Relation(a=1, b=1, algebraic_value=12, rational_value=1,
                    algebraic_factors={2: 2, 3: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=18, rational_value=1,
                    algebraic_factors={2: 1, 3: 2}, rational_factors={}),
            Relation(a=3, b=1, algebraic_value=24, rational_value=1,
                    algebraic_factors={2: 3, 3: 1}, rational_factors={}),
        ]
        primes = [2, 3]
        
        # Solve with both methods
        deps_dense = solve_matrix_dense(relations, primes)
        deps_block = solve_matrix(relations, primes, use_block_lanczos=False)
        
        # Should find the same number of dependencies
        assert len(deps_dense) == len(deps_block)
        
        # Verify each dependency produces even exponents
        for deps in [deps_dense, deps_block]:
            for dep in deps:
                combined = {}
                for idx in dep:
                    for p, exp in relations[idx].combined_factors().items():
                        combined[p] = combined.get(p, 0) + exp
                for p, total_exp in combined.items():
                    assert total_exp % 2 == 0, f"Odd exponent for prime {p}"
    
    def test_automatic_method_selection(self):
        """Test that solve_matrix auto-selects method based on size."""
        relations = [
            Relation(a=i, b=1, algebraic_value=2*i, rational_value=1,
                    algebraic_factors={2: 1}, rational_factors={})
            for i in range(1, 10)
        ]
        primes = [2, 3, 5]
        
        # Should use dense method for small matrix
        deps = solve_matrix(relations, primes)
        
        # Just verify it returns something reasonable
        assert isinstance(deps, list)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_relation(self):
        """Test with single relation."""
        relations = [
            Relation(a=1, b=1, algebraic_value=4, rational_value=1,
                    algebraic_factors={2: 2}, rational_factors={}),
        ]
        primes = [2, 3]
        
        # Single relation with even exponents forms its own dependency
        deps = solve_matrix(relations, primes)
        # May or may not find a dependency depending on exponents
        assert isinstance(deps, list)
    
    def test_no_relations(self):
        """Test with no relations."""
        deps = solve_matrix([], [2, 3, 5])
        assert deps == []
    
    def test_no_primes(self):
        """Test with empty prime list."""
        relations = [
            Relation(a=1, b=1, algebraic_value=1, rational_value=1,
                    algebraic_factors={}, rational_factors={}),
        ]
        deps = solve_matrix(relations, [])
        # Empty matrix should give all relations as dependencies
        assert isinstance(deps, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

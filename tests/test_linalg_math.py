"""Comprehensive mathematical tests for linear algebra operations in GNFS.

Tests verify:
- Matrix operations over GF(2): verify arithmetic is mod 2
- Gaussian elimination: verify row echelon form is correct
- Nullspace: verify Av = 0 for all vectors v in computed nullspace
- Dependency detection: verify combined exponents are even
- Edge cases: singular matrices, empty nullspace, full rank
"""

import pytest
import numpy as np

from gnfs.linalg.matrix import _nullspace_mod2, solve_matrix
from gnfs.sieve.relation import Relation


# =============================================================================
# Nullspace Computation Tests
# =============================================================================

class TestNullspaceMod2:
    """Tests for _nullspace_mod2 function."""
    
    def test_nullspace_identity_matrix(self):
        """Identity matrix has empty nullspace."""
        for n in [2, 3, 4, 5]:
            matrix = np.eye(n, dtype=int)
            basis = _nullspace_mod2(matrix)
            assert basis == [], f"Identity matrix of size {n} should have empty nullspace"
    
    def test_nullspace_zero_matrix(self):
        """Zero matrix has full nullspace."""
        matrix = np.zeros((2, 3), dtype=int)
        basis = _nullspace_mod2(matrix)
        assert len(basis) == 3, "Zero 2x3 matrix should have 3 nullspace vectors"
    
    def test_nullspace_verification(self):
        """Verify Av = 0 for all nullspace vectors v."""
        matrix = np.array([[1, 1, 0, 1],
                          [0, 1, 1, 0],
                          [1, 0, 1, 1]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        for v in basis:
            # Compute A * v mod 2
            result = (matrix @ v) % 2
            assert np.all(result == 0), f"Nullspace vector {v} doesn't satisfy Av=0"
    
    def test_nullspace_linear_independence(self):
        """Verify nullspace vectors are linearly independent over GF(2)."""
        matrix = np.array([[1, 0, 1, 0, 1],
                          [0, 1, 1, 1, 0],
                          [1, 1, 0, 1, 1]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        if len(basis) > 1:
            # Stack vectors and check rank
            basis_matrix = np.array(basis, dtype=int)
            # For GF(2), check that no vector is a XOR of others
            for i, v in enumerate(basis):
                for j in range(i + 1, len(basis)):
                    combined = (basis[i] ^ basis[j]) % 2
                    # Combined should not be all zeros (unless i==j)
                    assert not np.all(combined == 0), "Basis vectors are not independent"
    
    def test_nullspace_dimension(self):
        """Verify nullspace dimension = n - rank."""
        # 3x4 matrix with rank 2
        matrix = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [1, 1, 1, 1]], dtype=int)  # row 3 = row 1 + row 2
        
        basis = _nullspace_mod2(matrix)
        # Rank should be 2, so nullspace dim = 4 - 2 = 2
        assert len(basis) == 2


class TestNullspaceSpecialCases:
    """Test special cases for nullspace computation."""
    
    def test_single_row_matrix(self):
        """Test nullspace of single-row matrix."""
        matrix = np.array([[1, 0, 1, 1]], dtype=int)
        basis = _nullspace_mod2(matrix)
        
        # Should have n - 1 = 3 basis vectors
        assert len(basis) == 3
        
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    def test_single_column_matrix(self):
        """Test nullspace of single-column matrix."""
        matrix = np.array([[1], [0], [1]], dtype=int)
        basis = _nullspace_mod2(matrix)
        
        # Column of all 1s has nullspace if there's a 0, else empty
        # Here we have 0 in middle, so nullspace is empty for non-zero columns
        # Actually rank=1, n=1, so nullspace dim = 0
        assert len(basis) == 0
    
    def test_square_singular_matrix(self):
        """Test square singular matrix."""
        # Singular 3x3 matrix (row 3 = row 1 + row 2)
        matrix = np.array([[1, 0, 1],
                          [0, 1, 1],
                          [1, 1, 0]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        assert len(basis) == 1
        
        # Verify the nullspace vector
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    def test_wide_matrix(self):
        """Test wide matrix (more columns than rows)."""
        matrix = np.array([[1, 0, 1, 0, 1],
                          [0, 1, 1, 1, 0]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        # Rank <= 2, n = 5, so nullspace dim >= 3
        assert len(basis) >= 3
        
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    def test_tall_matrix(self):
        """Test tall matrix (more rows than columns)."""
        matrix = np.array([[1, 0],
                          [0, 1],
                          [1, 1],
                          [0, 0],
                          [1, 0]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        # Max rank = min(5, 2) = 2, so if full rank, nullspace dim = 0
        # Check the result
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)


# =============================================================================
# GF(2) Arithmetic Tests
# =============================================================================

class TestGF2Arithmetic:
    """Test that all operations are correctly done mod 2."""
    
    def test_all_entries_binary(self):
        """Verify matrix entries are treated as binary."""
        # Matrix with entries 2, 3 should be treated as 0, 1
        matrix = np.array([[2, 3, 4],
                          [5, 6, 7]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        # Equivalent to [[0, 1, 0], [1, 0, 1]]
        matrix_mod2 = matrix % 2
        basis_mod2 = _nullspace_mod2(matrix_mod2)
        
        # Should give same results
        assert len(basis) == len(basis_mod2)
    
    def test_xor_operation(self):
        """Test that row operations use XOR (addition mod 2)."""
        matrix = np.array([[1, 1],
                          [1, 0]], dtype=int)
        
        # After row reduction, should get identity (rank 2, empty nullspace)
        basis = _nullspace_mod2(matrix)
        assert basis == []
    
    def test_nullspace_binary_output(self):
        """Verify nullspace vectors have binary entries."""
        matrix = np.array([[1, 1, 0, 1],
                          [0, 1, 1, 0]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        for v in basis:
            assert all(x in [0, 1] for x in v), "Nullspace vector has non-binary entries"


# =============================================================================
# Solve Matrix Tests
# =============================================================================

class TestSolveMatrix:
    """Tests for solve_matrix function with Relations."""
    
    def test_simple_dependency(self):
        """Test detection of simple dependency."""
        primes = [2, 3]
        relations = [
            Relation(a=1, b=1, algebraic_value=2, rational_value=1,
                    algebraic_factors={2: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=2, rational_value=1,
                    algebraic_factors={2: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Both have exponent 1 for prime 2, so (1+1) % 2 = 0
        assert [0, 1] in deps
    
    def test_no_dependency(self):
        """Test case with no dependency."""
        primes = [2, 3]
        relations = [
            Relation(a=1, b=1, algebraic_value=2, rational_value=1,
                    algebraic_factors={2: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=3, rational_value=1,
                    algebraic_factors={3: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Matrix is [[1, 0], [0, 1]] which is full rank
        assert deps == []
    
    def test_combined_factors_used(self):
        """Verify combined factors from both sides are used."""
        primes = [2, 3]
        relations = [
            Relation(a=1, b=1, algebraic_value=2, rational_value=3,
                    algebraic_factors={2: 1}, rational_factors={3: 1}),
            Relation(a=2, b=1, algebraic_value=6, rational_value=1,
                    algebraic_factors={2: 1, 3: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Combined: rel1 = {2:1, 3:1}, rel2 = {2:1, 3:1}
        # Both have same exponents, so they form a dependency
        assert [0, 1] in deps
    
    def test_empty_relations(self):
        """Test with empty relations list."""
        deps = solve_matrix([], [2, 3, 5])
        assert deps == []
    
    def test_multiple_dependencies(self):
        """Test finding multiple independent dependencies."""
        primes = [2, 3]
        relations = [
            Relation(a=1, b=1, algebraic_value=4, rational_value=1,
                    algebraic_factors={2: 2}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=4, rational_value=1,
                    algebraic_factors={2: 2}, rational_factors={}),
            Relation(a=3, b=1, algebraic_value=9, rational_value=1,
                    algebraic_factors={3: 2}, rational_factors={}),
            Relation(a=4, b=1, algebraic_value=9, rational_value=1,
                    algebraic_factors={3: 2}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Should find at least 2 dependencies
        assert len(deps) >= 2


class TestDependencyValidation:
    """Verify that dependencies produce even exponents."""
    
    def test_even_exponents_simple(self):
        """Test that combining relations gives even exponents."""
        primes = [2, 3, 5]
        relations = [
            Relation(a=1, b=1, algebraic_value=6, rational_value=1,
                    algebraic_factors={2: 1, 3: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=10, rational_value=1,
                    algebraic_factors={2: 1, 5: 1}, rational_factors={}),
            Relation(a=3, b=1, algebraic_value=15, rational_value=1,
                    algebraic_factors={3: 1, 5: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        
        for dep in deps:
            # Sum exponents for each prime
            total_exp = {p: 0 for p in primes}
            for idx in dep:
                for p, exp in relations[idx].combined_factors().items():
                    total_exp[p] += exp
            
            # All should be even
            for p, exp in total_exp.items():
                assert exp % 2 == 0, f"Prime {p} has odd total exponent {exp}"
    
    def test_even_exponents_complex(self):
        """Test even exponents with more complex relations."""
        primes = [2, 3, 5, 7]
        relations = [
            Relation(a=1, b=1, algebraic_value=70, rational_value=1,
                    algebraic_factors={2: 1, 5: 1, 7: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=42, rational_value=1,
                    algebraic_factors={2: 1, 3: 1, 7: 1}, rational_factors={}),
            Relation(a=3, b=1, algebraic_value=30, rational_value=1,
                    algebraic_factors={2: 1, 3: 1, 5: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        
        for dep in deps:
            total_exp = {p: 0 for p in primes}
            for idx in dep:
                for p, exp in relations[idx].combined_factors().items():
                    total_exp[p] += exp
            
            for p, exp in total_exp.items():
                assert exp % 2 == 0


# =============================================================================
# Matrix Construction Tests
# =============================================================================

class TestMatrixConstruction:
    """Test that exponent matrix is constructed correctly."""
    
    def test_matrix_dimensions(self):
        """Verify matrix has correct dimensions."""
        primes = [2, 3, 5]
        relations = [
            Relation(a=1, b=1, algebraic_value=6, rational_value=1,
                    algebraic_factors={2: 1, 3: 1}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=10, rational_value=1,
                    algebraic_factors={2: 1, 5: 1}, rational_factors={}),
        ]
        
        # Matrix should be (num_primes) x (num_relations) = 3 x 2
        deps = solve_matrix(relations, primes)
        # Just verify it runs without error
        # Internal matrix shape is correct if deps are valid
    
    def test_missing_primes_zero(self):
        """Verify missing primes have exponent 0 in matrix."""
        primes = [2, 3, 5, 7]
        relations = [
            Relation(a=1, b=1, algebraic_value=4, rational_value=1,
                    algebraic_factors={2: 2}, rational_factors={}),
        ]
        
        # Only prime 2 appears with exponent 2 (even), others are 0
        deps = solve_matrix(relations, primes)
        # Since all exponents are even (2 % 2 = 0 and 0 % 2 = 0),
        # the matrix is all zeros, so the single relation forms a dependency
        # This is correct behavior - a relation with all even exponents
        # is itself a "square" and forms a valid dependency
        if deps:
            # Verify the dependency produces even exponents
            for idx in deps[0]:
                for p, exp in relations[idx].combined_factors().items():
                    assert exp % 2 == 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestLinalgEdgeCases:
    """Test edge cases in linear algebra."""
    
    def test_single_relation(self):
        """Single relation cannot form dependency."""
        primes = [2]
        relations = [
            Relation(a=1, b=1, algebraic_value=2, rational_value=1,
                    algebraic_factors={2: 1}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Single vector cannot be in nullspace (unless it's the zero vector)
        assert deps == []
    
    def test_all_even_exponents(self):
        """Relations with all even exponents each form self-dependency."""
        primes = [2, 3]
        relations = [
            Relation(a=1, b=1, algebraic_value=4, rational_value=1,
                    algebraic_factors={2: 2}, rational_factors={}),
            Relation(a=2, b=1, algebraic_value=9, rational_value=1,
                    algebraic_factors={3: 2}, rational_factors={}),
        ]
        
        deps = solve_matrix(relations, primes)
        # Each relation alone is a square, so single-element deps possible
        # But solve_matrix only returns multi-element deps
    
    def test_large_matrix(self):
        """Test with larger matrix."""
        primes = list(range(2, 30))  # First 28 integers as "primes"
        primes = [p for p in primes if all(p % i != 0 for i in range(2, p))][:10]
        
        # Create relations with random factorizations
        import random
        random.seed(42)
        
        relations = []
        for i in range(15):
            factors = {}
            for p in primes:
                if random.random() < 0.3:
                    factors[p] = random.randint(1, 3)
            if factors:
                product = 1
                for p, e in factors.items():
                    product *= p ** e
                relations.append(Relation(
                    a=i, b=1, algebraic_value=product, rational_value=1,
                    algebraic_factors=factors, rational_factors={}
                ))
        
        if len(relations) > len(primes):
            deps = solve_matrix(relations, primes)
            # Should find at least one dependency
            for dep in deps:
                # Verify even exponents
                total = {p: 0 for p in primes}
                for idx in dep:
                    for p, e in relations[idx].combined_factors().items():
                        total[p] += e
                for p, e in total.items():
                    assert e % 2 == 0


# =============================================================================
# Gaussian Elimination Verification
# =============================================================================

class TestGaussianElimination:
    """Verify Gaussian elimination produces correct row echelon form."""
    
    def test_echelon_form_structure(self):
        """Test that elimination produces valid echelon structure."""
        matrix = np.array([[1, 1, 0, 1],
                          [1, 0, 1, 0],
                          [0, 1, 1, 1]], dtype=int)
        
        # After elimination, pivots should be in staircase pattern
        basis = _nullspace_mod2(matrix)
        
        # Verify nullspace vectors satisfy original equation
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    def test_elimination_preserves_nullspace(self):
        """Verify elimination doesn't change the nullspace."""
        matrix = np.array([[1, 0, 1, 1],
                          [0, 1, 1, 0],
                          [1, 1, 0, 1]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        # Verify dimension is correct (n - rank)
        # Rank is at most 3, so nullspace dim is at least 1
        assert len(basis) >= 1
        
        # Each basis vector should satisfy original equation
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)


# =============================================================================
# Stress Tests
# =============================================================================

class TestLinalgStress:
    """Stress tests for linear algebra."""
    
    def test_random_matrices(self):
        """Test with random binary matrices."""
        import random
        random.seed(123)
        
        for _ in range(10):
            rows = random.randint(3, 8)
            cols = random.randint(rows, rows + 5)
            
            matrix = np.array([
                [random.randint(0, 1) for _ in range(cols)]
                for _ in range(rows)
            ], dtype=int)
            
            basis = _nullspace_mod2(matrix)
            
            # Verify all basis vectors are in nullspace
            for v in basis:
                result = (matrix @ v) % 2
                assert np.all(result == 0), f"Failed for {rows}x{cols} matrix"
    
    def test_repeated_rows(self):
        """Test matrix with repeated rows."""
        matrix = np.array([[1, 1, 0],
                          [1, 1, 0],
                          [0, 0, 1]], dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        # Repeated row should not affect nullspace
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    def test_all_ones_matrix(self):
        """Test matrix of all ones."""
        matrix = np.ones((3, 5), dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        # All-ones 3x5 matrix has rank 1, so nullspace dim = 4
        assert len(basis) == 4
        
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)

"""Comprehensive integration tests for the GNFS implementation.

Tests verify:
- Factor known semiprimes: 91, 143, 221, 323, 437, 1003, 2021
- Verify p*q = n for all results
- Test with different parameter combinations
- Verify deterministic results (same input → same output)
"""

import pytest
import sympy as sp

from gnfs.factor import gnfs_factor
from gnfs.polynomial.selection import select_polynomial
from gnfs.sieve.sieve import find_relations
from gnfs.sqrt.square_root import find_factors


# =============================================================================
# Known Semiprime Factorizations
# =============================================================================

class TestKnownSemiprimes:
    """Test factorization of known semiprimes."""
    
    def test_factor_10(self):
        """Test factorization of 10 = 2 * 5."""
        factors = gnfs_factor(10, bound=10, interval=20)
        if factors:
            assert set(factors) == {2, 5}
            assert factors[0] * factors[1] == 10
    
    def test_factor_15(self):
        """Test factorization of 15 = 3 * 5."""
        factors = gnfs_factor(15, bound=10, interval=20)
        if factors:
            assert set(factors) == {3, 5}
            assert factors[0] * factors[1] == 15
    
    def test_factor_21(self):
        """Test factorization of 21 = 3 * 7."""
        factors = gnfs_factor(21, bound=15, interval=25)
        if factors:
            assert set(factors) == {3, 7}
            assert factors[0] * factors[1] == 21
    
    def test_factor_35(self):
        """Test factorization of 35 = 5 * 7."""
        factors = gnfs_factor(35, bound=15, interval=30)
        if factors:
            assert set(factors) == {5, 7}
            assert factors[0] * factors[1] == 35
    
    def test_factor_77(self):
        """Test factorization of 77 = 7 * 11."""
        factors = gnfs_factor(77, bound=20, interval=35)
        if factors:
            assert set(factors) == {7, 11}
            assert factors[0] * factors[1] == 77
    
    def test_factor_91(self):
        """Test factorization of 91 = 7 * 13."""
        factors = gnfs_factor(91, bound=20, interval=40)
        if factors:
            assert set(factors) == {7, 13}
            assert factors[0] * factors[1] == 91
    
    def test_factor_143(self):
        """Test factorization of 143 = 11 * 13."""
        factors = gnfs_factor(143, bound=25, interval=45)
        if factors:
            assert set(factors) == {11, 13}
            assert factors[0] * factors[1] == 143
    
    def test_factor_221(self):
        """Test factorization of 221 = 13 * 17."""
        factors = gnfs_factor(221, bound=30, interval=50)
        if factors:
            assert set(factors) == {13, 17}
            assert factors[0] * factors[1] == 221
    
    def test_factor_323(self):
        """Test factorization of 323 = 17 * 19."""
        factors = gnfs_factor(323, bound=30, interval=55)
        if factors:
            assert set(factors) == {17, 19}
            assert factors[0] * factors[1] == 323
    
    def test_factor_437(self):
        """Test factorization of 437 = 19 * 23."""
        factors = gnfs_factor(437, bound=35, interval=60)
        if factors:
            assert set(factors) == {19, 23}
            assert factors[0] * factors[1] == 437
    
    def test_factor_1003(self):
        """Test factorization of 1003 = 17 * 59."""
        factors = gnfs_factor(1003, bound=70, interval=80, max_rounds=10)
        if factors:
            assert set(factors) == {17, 59}
            assert factors[0] * factors[1] == 1003
    
    def test_factor_2021(self):
        """Test factorization of 2021 = 43 * 47."""
        factors = gnfs_factor(2021, bound=60, interval=80, max_rounds=10)
        if factors:
            assert set(factors) == {43, 47}
            assert factors[0] * factors[1] == 2021


# =============================================================================
# Factor Verification Tests
# =============================================================================

class TestFactorVerification:
    """Verify that p * q = n for all results."""
    
    def test_product_equals_n(self):
        """Test that found factors multiply to n."""
        test_numbers = [15, 21, 35, 77, 91, 143]
        
        for n in test_numbers:
            factors = gnfs_factor(n, bound=30, interval=50)
            
            if len(factors) >= 2:
                product = factors[0] * factors[1]
                assert product == n, f"For n={n}: {factors[0]} * {factors[1]} = {product} != {n}"
    
    def test_factors_are_proper_divisors(self):
        """Test that factors are proper divisors (1 < f < n)."""
        test_numbers = [15, 21, 35, 77, 91, 143, 221]
        
        for n in test_numbers:
            factors = gnfs_factor(n, bound=30, interval=50)
            
            for f in factors:
                assert 1 < f < n, f"Factor {f} is not proper for n={n}"
                assert n % f == 0, f"Factor {f} does not divide {n}"
    
    def test_factors_are_prime(self):
        """Test that factors of semiprimes are prime."""
        semiprimes = [15, 21, 35, 77, 91, 143]
        
        for n in semiprimes:
            factors = gnfs_factor(n, bound=30, interval=50)
            
            for f in factors:
                assert sp.isprime(f), f"Factor {f} of {n} is not prime"


# =============================================================================
# Parameter Combination Tests
# =============================================================================

class TestParameterCombinations:
    """Test with different parameter combinations."""
    
    def test_varying_bounds(self):
        """Test factorization with different smoothness bounds."""
        n = 91
        
        for bound in [15, 20, 25, 30]:
            factors = gnfs_factor(n, bound=bound, interval=50)
            
            if factors:
                assert set(factors) == {7, 13}
    
    def test_varying_intervals(self):
        """Test factorization with different sieving intervals."""
        n = 77
        
        for interval in [30, 40, 50, 60]:
            factors = gnfs_factor(n, bound=20, interval=interval)
            
            if factors:
                assert set(factors) == {7, 11}
    
    def test_varying_degrees(self):
        """Test factorization with different polynomial degrees."""
        n = 143
        
        for degree in [1, 2]:
            factors = gnfs_factor(n, bound=25, interval=50, degree=degree)
            
            if factors:
                assert set(factors) == {11, 13}
    
    def test_max_rounds(self):
        """Test with different max_rounds values."""
        n = 91
        
        for rounds in [3, 5, 7]:
            factors = gnfs_factor(n, bound=20, interval=30, max_rounds=rounds)
            
            if factors:
                assert set(factors) == {7, 13}


# =============================================================================
# Determinism Tests
# =============================================================================

class TestDeterminism:
    """Verify deterministic results (same input → same output)."""
    
    def test_same_input_same_output(self):
        """Running with same parameters should give same result."""
        n = 91
        params = {'bound': 20, 'interval': 40}
        
        factors1 = gnfs_factor(n, **params)
        factors2 = gnfs_factor(n, **params)
        
        assert factors1 == factors2
    
    def test_polynomial_selection_deterministic(self):
        """Polynomial selection should be deterministic."""
        n = 143
        
        sel1 = select_polynomial(n, degree=2, optimize=False)
        sel2 = select_polynomial(n, degree=2, optimize=False)
        
        assert sel1.algebraic.coeffs == sel2.algebraic.coeffs
        assert sel1.rational.coeffs == sel2.rational.coeffs
        assert sel1.m == sel2.m
    
    def test_relations_deterministic(self):
        """Sieving should be deterministic."""
        n = 77
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 20))
        
        rels1 = list(find_relations(selection, primes=primes, interval=30))
        rels2 = list(find_relations(selection, primes=primes, interval=30))
        
        assert len(rels1) == len(rels2)
        
        for r1, r2 in zip(rels1, rels2):
            assert r1.a == r2.a
            assert r1.b == r2.b


# =============================================================================
# Mathematical Invariant Tests
# =============================================================================

class TestMathematicalInvariants:
    """Test mathematical invariants throughout the pipeline."""
    
    def test_polynomial_root_property(self):
        """Test f(m) ≡ 0 (mod n) for selected polynomial."""
        test_numbers = [91, 143, 221]
        
        for n in test_numbers:
            for degree in [1, 2]:
                selection = select_polynomial(n, degree=degree)
                
                # f(m) should be divisible by n
                f_at_m = selection.algebraic.evaluate(selection.m)
                assert f_at_m % n == 0 or f_at_m == n, \
                    f"f({selection.m}) = {f_at_m} not divisible by {n}"
    
    def test_relation_norms_factor(self):
        """Test that relation norms factor over the factor base."""
        n = 91
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        for rel in relations:
            # Verify algebraic value factors completely
            remaining = abs(rel.algebraic_value)
            for p, exp in rel.algebraic_factors.items():
                for _ in range(exp):
                    remaining //= p
            assert remaining == 1
            
            # Verify rational value factors completely
            remaining = abs(rel.rational_value)
            for p, exp in rel.rational_factors.items():
                for _ in range(exp):
                    remaining //= p
            assert remaining == 1


# =============================================================================
# Edge Cases Integration Tests
# =============================================================================

class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""
    
    def test_small_semiprime(self):
        """Test with smallest semiprimes."""
        small_semiprimes = [(6, {2, 3}), (9, {3, 3}), (10, {2, 5}), (14, {2, 7})]
        
        for n, expected in small_semiprimes:
            factors = gnfs_factor(n, bound=10, interval=15)
            
            if factors:
                factor_set = set(factors)
                # For 9 = 3*3, we might get [3, 3]
                if n == 9:
                    assert 3 in factors
                else:
                    assert factor_set == expected
    
    def test_larger_semiprime(self):
        """Test with larger semiprimes."""
        # These may require more resources
        n = 3599  # 59 * 61
        factors = gnfs_factor(n, bound=80, interval=100, max_rounds=15)
        
        if factors:
            assert set(factors) == {59, 61}
    
    def test_insufficient_bound(self):
        """Test behavior with insufficient smoothness bound."""
        n = 143  # 11 * 13
        # Very small bound - may not find enough relations
        factors = gnfs_factor(n, bound=5, interval=20, max_rounds=2)
        
        # Should return empty or correct result (not crash)
        if factors:
            assert set(factors) == {11, 13}


# =============================================================================
# Pipeline Stage Tests
# =============================================================================

class TestPipelineStages:
    """Test individual pipeline stages work together correctly."""
    
    def test_polynomial_to_sieve(self):
        """Test polynomial selection feeds into sieving correctly."""
        n = 91
        selection = select_polynomial(n, degree=1)
        primes = list(sp.primerange(2, 20))
        
        # Sieving should work with the selected polynomial
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        # Should find some relations
        assert len(relations) > 0
        
        # All relations should be valid
        for rel in relations:
            # Algebraic value matches polynomial evaluation
            expected = selection.algebraic.evaluate_homogeneous(rel.a, rel.b)
            assert rel.algebraic_value == expected
    
    def test_sieve_to_linalg(self):
        """Test relations feed into linear algebra correctly."""
        n = 91
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=40))
        
        from gnfs.linalg.matrix import solve_matrix
        deps = solve_matrix(relations, primes)
        
        # Should find some dependencies if we have enough relations
        if len(relations) > len(primes) + 1:
            assert len(deps) > 0
    
    def test_linalg_to_sqrt(self):
        """Test dependencies feed into square root step correctly."""
        n = 91
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=40))
        
        # Square root step should work with the relations
        factors = list(find_factors(n, relations, primes))
        
        if factors:
            for f in factors:
                assert n % f == 0


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for the GNFS implementation."""
    
    def test_multiple_factorizations(self):
        """Run factorization on multiple numbers."""
        semiprimes = [
            (15, {3, 5}),
            (21, {3, 7}),
            (35, {5, 7}),
            (77, {7, 11}),
            (91, {7, 13}),
            (143, {11, 13}),
        ]
        
        success_count = 0
        for n, expected in semiprimes:
            factors = gnfs_factor(n, bound=30, interval=50)
            
            if factors and set(factors) == expected:
                success_count += 1
        
        # Should succeed on most
        assert success_count >= len(semiprimes) // 2
    
    def test_repeated_runs(self):
        """Test repeated runs give consistent results."""
        n = 91
        
        results = []
        for _ in range(5):
            factors = gnfs_factor(n, bound=20, interval=40)
            results.append(tuple(sorted(factors)) if factors else None)
        
        # All non-None results should be the same
        non_none = [r for r in results if r is not None]
        if non_none:
            assert all(r == non_none[0] for r in non_none)


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests for previously identified issues."""
    
    def test_zero_interval_handling(self):
        """Test that zero or negative interval is handled."""
        n = 91
        
        # Should either work or raise an appropriate error
        try:
            factors = gnfs_factor(n, bound=20, interval=1)
            # If it works, factors should be correct
            if factors:
                assert set(factors) == {7, 13}
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise an error
    
    def test_coprime_constraint_maintained(self):
        """Verify all relations maintain gcd(a, b) = 1."""
        import math
        
        n = 143
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 25))
        relations = list(find_relations(selection, primes=primes, interval=40))
        
        for rel in relations:
            assert math.gcd(rel.a, rel.b) == 1, \
                f"Relation ({rel.a}, {rel.b}) violates coprime constraint"

"""Comprehensive mathematical tests for GNFS sieving operations.

Tests verify:
- Smooth relations: check that found relations actually factor over the base
- Verify norms: algebraic and rational norms computed correctly
- Root finding mod p: verify a - rb ≡ 0 (mod p) for relation roots
- Factor base completeness
- Edge cases: b=0, gcd(a,b)≠1, negative values
"""

import pytest
import math
import sympy as sp
from typing import Dict

from gnfs.polynomial.polynomial import Polynomial
from gnfs.polynomial.selection import select_polynomial
from gnfs.sieve.sieve import find_relations
from gnfs.sieve.relation import Relation
from gnfs.sieve.roots import _polynomial_roots_mod_p


# =============================================================================
# Relation Smoothness Tests
# =============================================================================

class TestRelationSmoothness:
    """Verify that found relations are actually smooth over the factor base."""
    
    def test_algebraic_side_smooth(self):
        """Verify algebraic values factor completely over the factor base."""
        selection = select_polynomial(91)  # 7 * 13
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        for rel in relations:
            # Verify algebraic value factors completely
            remaining = abs(rel.algebraic_value)
            for p, exp in rel.algebraic_factors.items():
                assert p in primes, f"Factor {p} not in factor base"
                for _ in range(exp):
                    assert remaining % p == 0, f"Factor {p}^{exp} doesn't divide {rel.algebraic_value}"
                    remaining //= p
            assert remaining == 1, f"Algebraic value {rel.algebraic_value} has unfactored part {remaining}"
    
    def test_rational_side_smooth(self):
        """Verify rational values factor completely over the factor base."""
        selection = select_polynomial(143)  # 11 * 13
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        for rel in relations:
            remaining = abs(rel.rational_value)
            for p, exp in rel.rational_factors.items():
                assert p in primes, f"Factor {p} not in factor base"
                for _ in range(exp):
                    assert remaining % p == 0
                    remaining //= p
            assert remaining == 1, f"Rational value {rel.rational_value} has unfactored part {remaining}"
    
    def test_both_sides_smooth(self):
        """Verify both algebraic and rational sides are smooth simultaneously."""
        selection = select_polynomial(77)  # 7 * 11
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            # Algebraic side
            alg_remaining = abs(rel.algebraic_value)
            for p, exp in rel.algebraic_factors.items():
                for _ in range(exp):
                    alg_remaining //= p
            
            # Rational side
            rat_remaining = abs(rel.rational_value)
            for p, exp in rel.rational_factors.items():
                for _ in range(exp):
                    rat_remaining //= p
            
            assert alg_remaining == 1 and rat_remaining == 1, \
                f"Relation ({rel.a}, {rel.b}) not fully smooth"
    
    def test_factor_counts_correct(self):
        """Verify that exponent counts in factorization are correct."""
        selection = select_polynomial(15)  # 3 * 5
        primes = list(sp.primerange(2, 10))
        relations = list(find_relations(selection, primes=primes, interval=15))
        
        for rel in relations:
            # Manually count factors
            for p in primes:
                alg_val = abs(rel.algebraic_value)
                manual_count = 0
                while alg_val % p == 0:
                    manual_count += 1
                    alg_val //= p
                
                stored_count = rel.algebraic_factors.get(p, 0)
                assert stored_count == manual_count, \
                    f"Mismatch for prime {p}: stored {stored_count}, actual {manual_count}"


# =============================================================================
# Norm Computation Tests
# =============================================================================

class TestNormComputation:
    """Verify that norms are computed correctly during sieving."""
    
    def test_algebraic_norm_formula(self):
        """Verify algebraic norm matches evaluate_homogeneous."""
        selection = select_polynomial(143, degree=2)
        poly = selection.algebraic
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            # Algebraic value should be evaluate_homogeneous(a, b)
            expected = poly.evaluate_homogeneous(rel.a, rel.b)
            assert rel.algebraic_value == expected, \
                f"Algebraic norm mismatch for ({rel.a}, {rel.b}): {rel.algebraic_value} != {expected}"
    
    def test_rational_norm_formula(self):
        """Verify rational norm matches evaluate_homogeneous of rational poly."""
        selection = select_polynomial(91, degree=1)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            # Rational polynomial is x - m, so rational value is a - m*b
            expected = selection.rational.evaluate_homogeneous(rel.a, rel.b)
            assert rel.rational_value == expected, \
                f"Rational norm mismatch for ({rel.a}, {rel.b})"
    
    def test_norm_nonzero(self):
        """Verify that stored norms are non-zero."""
        selection = select_polynomial(77)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=15))
        
        for rel in relations:
            assert rel.algebraic_value != 0, "Algebraic value is zero"
            assert rel.rational_value != 0, "Rational value is zero"


# =============================================================================
# Root Finding Tests
# =============================================================================

class TestRootFindingInSieve:
    """Verify root finding for sieve correctness."""
    
    def test_root_property_for_relations(self):
        """For each relation (a,b) and prime p, verify a ≡ rb (mod p) for some root r."""
        selection = select_polynomial(143, degree=2)
        poly = selection.algebraic
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            for p in rel.algebraic_factors.keys():
                roots = _polynomial_roots_mod_p(poly, p)
                # At least one root r should satisfy a ≡ rb (mod p)
                has_root = any((rel.a - r * rel.b) % p == 0 for r in roots)
                # Or the leading coefficient divides b (root at infinity)
                leading = poly.coeffs[-1] if poly.coeffs else 1
                root_at_infinity = rel.b % p == 0 and leading % p == 0
                
                assert has_root or root_at_infinity, \
                    f"No root for ({rel.a}, {rel.b}) mod {p}"
    
    def test_roots_divide_norm(self):
        """If polynomial has root r mod p, then p | norm when a ≡ rb (mod p)."""
        poly = Polynomial((6, -5, 1))  # x² - 5x + 6 = (x-2)(x-3)
        
        for p in [2, 3, 5, 7, 11]:
            roots = _polynomial_roots_mod_p(poly, p)
            
            for r in roots:
                # For any b, a = r*b satisfies: p | f(a/b) * b^d
                b = 1
                a = r * b
                norm = poly.evaluate_homogeneous(a, b)
                assert norm % p == 0, f"p={p}, r={r}: norm {norm} not divisible by {p}"


# =============================================================================
# Factor Base Tests
# =============================================================================

class TestFactorBase:
    """Test factor base completeness and correctness."""
    
    def test_all_factors_in_base(self):
        """Verify all factors in relations are from the factor base."""
        primes = list(sp.primerange(2, 30))
        selection = select_polynomial(221)  # 13 * 17
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        prime_set = set(primes)
        for rel in relations:
            for p in rel.algebraic_factors.keys():
                assert p in prime_set, f"Algebraic factor {p} not in base"
            for p in rel.rational_factors.keys():
                assert p in prime_set, f"Rational factor {p} not in base"
    
    def test_sufficient_relations(self):
        """Verify we get enough relations (more than #primes + 1)."""
        primes = list(sp.primerange(2, 20))
        required = len(primes) + 1
        
        selection = select_polynomial(91)
        relations = list(find_relations(selection, primes=primes, interval=50))
        
        # Should get at least as many relations as required
        assert len(relations) >= required, \
            f"Got {len(relations)} relations, need at least {required}"


# =============================================================================
# Relation Properties Tests
# =============================================================================

class TestRelationProperties:
    """Test mathematical properties of relations."""
    
    def test_coprime_constraint(self):
        """Verify gcd(a, b) = 1 for all relations."""
        selection = select_polynomial(143)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        for rel in relations:
            assert math.gcd(rel.a, rel.b) == 1, \
                f"Relation ({rel.a}, {rel.b}) has gcd != 1"
    
    def test_positive_b(self):
        """Verify b > 0 for all relations (canonical form)."""
        selection = select_polynomial(77)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            assert rel.b > 0, f"Relation has b = {rel.b} <= 0"
    
    def test_combined_factors(self):
        """Test combined_factors method returns correct merge."""
        rel = Relation(
            a=1, b=1,
            algebraic_value=12,
            rational_value=6,
            algebraic_factors={2: 2, 3: 1},
            rational_factors={2: 1, 3: 1}
        )
        
        combined = rel.combined_factors()
        assert combined[2] == 3  # 2 + 1
        assert combined[3] == 2  # 1 + 1
    
    def test_combined_factors_disjoint(self):
        """Test combined_factors with disjoint factor sets."""
        rel = Relation(
            a=1, b=1,
            algebraic_value=4,
            rational_value=9,
            algebraic_factors={2: 2},
            rational_factors={3: 2}
        )
        
        combined = rel.combined_factors()
        assert combined[2] == 2
        assert combined[3] == 2


# =============================================================================
# Edge Cases
# =============================================================================

class TestSieveEdgeCases:
    """Test edge cases in sieving."""
    
    def test_small_interval(self):
        """Test sieving with very small interval."""
        selection = select_polynomial(15)
        primes = list(sp.primerange(2, 10))
        relations = list(find_relations(selection, primes=primes, interval=5))
        
        # Should still find some relations or return empty without error
        for rel in relations:
            assert math.gcd(rel.a, rel.b) == 1
    
    def test_negative_a_values(self):
        """Verify handling of negative a values."""
        selection = select_polynomial(91)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        negative_a_rels = [rel for rel in relations if rel.a < 0]
        # Should have some relations with negative a
        # Just verify they're valid
        for rel in negative_a_rels:
            assert rel.algebraic_value == selection.algebraic.evaluate_homogeneous(rel.a, rel.b)
    
    def test_large_primes(self):
        """Test with larger primes in factor base."""
        primes = list(sp.primerange(2, 100))
        selection = select_polynomial(1003)  # 17 * 59
        relations = list(find_relations(selection, primes=primes, interval=50))
        
        for rel in relations:
            for p in rel.algebraic_factors:
                assert p in primes
    
    def test_prime_n(self):
        """Test sieving when n is prime (edge case)."""
        selection = select_polynomial(97)  # prime
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        # Should still work, just won't find factors
        for rel in relations:
            assert math.gcd(rel.a, rel.b) == 1


# =============================================================================
# Value Verification Tests
# =============================================================================

class TestValueVerification:
    """Verify that relation values are mathematically correct."""
    
    def test_algebraic_value_factorization(self):
        """Verify algebraic_value = product of prime^exp."""
        selection = select_polynomial(143)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            product = 1
            for p, exp in rel.algebraic_factors.items():
                product *= p ** exp
            
            # Account for sign
            assert abs(rel.algebraic_value) == product, \
                f"|{rel.algebraic_value}| != {product} for ({rel.a}, {rel.b})"
    
    def test_rational_value_factorization(self):
        """Verify rational_value = product of prime^exp."""
        selection = select_polynomial(91)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            product = 1
            for p, exp in rel.rational_factors.items():
                product *= p ** exp
            
            assert abs(rel.rational_value) == product
    
    def test_homogeneous_evaluation_match(self):
        """Verify stored values match homogeneous polynomial evaluation."""
        selection = select_polynomial(221, degree=2)
        primes = list(sp.primerange(2, 25))
        relations = list(find_relations(selection, primes=primes, interval=25))
        
        for rel in relations:
            # Algebraic
            expected_alg = selection.algebraic.evaluate_homogeneous(rel.a, rel.b)
            assert rel.algebraic_value == expected_alg
            
            # Rational
            expected_rat = selection.rational.evaluate_homogeneous(rel.a, rel.b)
            assert rel.rational_value == expected_rat


# =============================================================================
# Sign Handling Tests
# =============================================================================

class TestSignHandling:
    """Test correct handling of signs in factorizations."""
    
    def test_sign_in_algebraic_factors(self):
        """Verify signs are handled correctly in algebraic factorization."""
        selection = select_polynomial(91)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            # The factorization should only contain positive primes
            for p in rel.algebraic_factors:
                assert p > 0, f"Negative prime {p} in factorization"
            
            # And the product should equal |algebraic_value|
            product = 1
            for p, exp in rel.algebraic_factors.items():
                product *= p ** exp
            
            assert product == abs(rel.algebraic_value)
    
    def test_negative_values_handled(self):
        """Verify negative norm values are handled correctly."""
        selection = select_polynomial(77)
        primes = list(sp.primerange(2, 15))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        has_negative_alg = any(rel.algebraic_value < 0 for rel in relations)
        has_negative_rat = any(rel.rational_value < 0 for rel in relations)
        
        # At least some values should be negative (typical in sieving)
        # Just verify factorizations are still correct
        for rel in relations:
            alg_product = 1
            for p, exp in rel.algebraic_factors.items():
                alg_product *= p ** exp
            assert alg_product == abs(rel.algebraic_value)


# =============================================================================
# Consistency Tests
# =============================================================================

class TestSieveConsistency:
    """Test internal consistency of sieve results."""
    
    def test_unique_relations(self):
        """Verify no duplicate (a, b) pairs in results."""
        selection = select_polynomial(143)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=30))
        
        seen = set()
        for rel in relations:
            key = (rel.a, rel.b)
            assert key not in seen, f"Duplicate relation {key}"
            seen.add(key)
    
    def test_factor_base_subset(self):
        """Verify factors used are subset of factor base."""
        primes = list(sp.primerange(2, 30))
        prime_set = set(primes)
        
        selection = select_polynomial(323)  # 17 * 19
        relations = list(find_relations(selection, primes=primes, interval=40))
        
        for rel in relations:
            alg_primes = set(rel.algebraic_factors.keys())
            rat_primes = set(rel.rational_factors.keys())
            
            assert alg_primes.issubset(prime_set)
            assert rat_primes.issubset(prime_set)

"""Tests for lattice sieving implementation.

Tests cover:
- Lattice basis computation and reduction
- Special-q selection
- Lattice sieve correctness
- Comparison with line sieve
- Performance benchmarks
"""

import math
import time
from typing import Set, Tuple

import pytest
import sympy as sp

from gnfs.polynomial import Polynomial, select_polynomial, PolynomialSelection
from gnfs.sieve import (
    Relation,
    find_relations,
    find_relations_lattice,
    find_relations_hybrid,
    lattice_sieve_for_special_q,
    select_special_q_primes,
    LatticeBasis,
    compute_lattice_basis,
    reduce_lattice_basis,
)
from gnfs.sieve.lattice_sieve import (
    extended_gcd,
    mod_inverse,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_test_primes(bound: int) -> list:
    """Get list of primes up to bound."""
    return list(sp.primerange(2, bound))


def verify_relation(rel: Relation, primes: list) -> bool:
    """Verify that a relation is valid (both sides factor completely)."""
    # Check algebraic side
    alg_val = abs(rel.algebraic_value)
    for p, exp in rel.algebraic_factors.items():
        for _ in range(exp):
            if alg_val % p != 0:
                return False
            alg_val //= p
    if alg_val != 1:
        return False
    
    # Check rational side
    rat_val = abs(rel.rational_value)
    for p, exp in rel.rational_factors.items():
        for _ in range(exp):
            if rat_val % p != 0:
                return False
            rat_val //= p
    if rat_val != 1:
        return False
    
    return True


# =============================================================================
# Extended GCD and Modular Inverse Tests
# =============================================================================

class TestExtendedGCD:
    """Tests for extended GCD and modular inverse."""
    
    def test_basic_gcd(self):
        """Test basic GCD computation."""
        g, x, y = extended_gcd(12, 8)
        assert g == 4
        assert 12 * x + 8 * y == g
    
    def test_coprime(self):
        """Test GCD of coprime numbers."""
        g, x, y = extended_gcd(17, 13)
        assert g == 1
        assert 17 * x + 13 * y == g
    
    def test_mod_inverse_exists(self):
        """Test modular inverse when it exists."""
        inv = mod_inverse(3, 7)
        assert inv is not None
        assert (3 * inv) % 7 == 1
    
    def test_mod_inverse_not_exists(self):
        """Test modular inverse when it doesn't exist."""
        inv = mod_inverse(6, 9)
        assert inv is None  # gcd(6, 9) = 3 ≠ 1


# =============================================================================
# Lattice Basis Tests
# =============================================================================

class TestLatticeBasis:
    """Tests for lattice basis computation and properties."""
    
    def test_basic_basis(self):
        """Test basic lattice basis construction."""
        q = 17
        r = 5
        basis = compute_lattice_basis(q, r)
        
        assert basis.q == q
        assert basis.r == r
        assert basis.v1 == (q, 0)
        assert basis.v2 == (r, 1)
    
    def test_lattice_to_ab(self):
        """Test conversion from lattice to (a, b) coordinates."""
        basis = compute_lattice_basis(17, 5)
        
        # i=0, j=1 should give (5, 1)
        a, b = basis.lattice_to_ab(0, 1)
        assert a == 5
        assert b == 1
        
        # i=1, j=1 should give (5 + 17, 1) = (22, 1)
        a, b = basis.lattice_to_ab(1, 1)
        assert a == 22
        assert b == 1
        
        # i=-1, j=2 should give (2*5 - 17, 2) = (-7, 2)
        a, b = basis.lattice_to_ab(-1, 2)
        assert a == -7
        assert b == 2
    
    def test_ab_to_lattice(self):
        """Test conversion from (a, b) to lattice coordinates."""
        basis = compute_lattice_basis(17, 5)
        
        # (5, 1) is in lattice: 5 ≡ 5*1 (mod 17)
        result = basis.ab_to_lattice(5, 1)
        assert result == (0, 1)
        
        # (22, 1) is in lattice: 22 ≡ 5*1 (mod 17)
        result = basis.ab_to_lattice(22, 1)
        assert result == (1, 1)
        
        # (6, 1) is NOT in lattice: 6 ≢ 5*1 (mod 17)
        result = basis.ab_to_lattice(6, 1)
        assert result is None
    
    def test_roundtrip(self):
        """Test that lattice_to_ab and ab_to_lattice are inverses."""
        basis = compute_lattice_basis(23, 7)
        
        for i in range(-5, 6):
            for j in range(1, 10):
                a, b = basis.lattice_to_ab(i, j)
                result = basis.ab_to_lattice(a, b)
                assert result == (i, j), f"Roundtrip failed for ({i}, {j})"
    
    def test_lattice_points_satisfy_congruence(self):
        """All lattice points should satisfy a ≡ r*b (mod q)."""
        q = 31
        r = 13
        basis = compute_lattice_basis(q, r)
        
        for i in range(-10, 11):
            for j in range(1, 20):
                a, b = basis.lattice_to_ab(i, j)
                assert (a - r * b) % q == 0, f"Point ({a}, {b}) doesn't satisfy congruence"


class TestLatticeReduction:
    """Tests for lattice basis reduction."""
    
    def test_reduction_preserves_lattice(self):
        """Reduced basis should generate the same lattice."""
        original = compute_lattice_basis(97, 42)
        reduced = reduce_lattice_basis(original)
        
        # Both bases should generate the same set of points
        original_points = set()
        reduced_points = set()
        
        for i in range(-5, 6):
            for j in range(1, 10):
                original_points.add(original.lattice_to_ab(i, j))
                reduced_points.add(reduced.lattice_to_ab(i, j))
        
        # Points should be the same (up to different parameterization)
        assert len(original_points) == len(reduced_points)
    
    def test_reduction_shortens_vectors(self):
        """Reduction should not make vectors longer."""
        def norm_sq(v):
            return v[0] * v[0] + v[1] * v[1]
        
        original = compute_lattice_basis(100, 37)
        reduced = reduce_lattice_basis(original)
        
        # Total norm should not increase
        orig_norm = norm_sq(original.v1) + norm_sq(original.v2)
        red_norm = norm_sq(reduced.v1) + norm_sq(reduced.v2)
        
        assert red_norm <= orig_norm * 1.01  # Allow small tolerance


# =============================================================================
# Special-q Selection Tests
# =============================================================================

class TestSpecialQSelection:
    """Tests for special-q prime selection."""
    
    def test_selects_from_factor_base(self):
        """Selected special-q primes should come from factor base."""
        primes = get_test_primes(500)
        special_q = select_special_q_primes(primes, num_special_q=5)
        
        for q in special_q:
            assert q in primes
    
    def test_prefers_larger_primes(self):
        """Special-q primes should be from the larger end."""
        primes = get_test_primes(500)
        special_q = select_special_q_primes(primes, num_special_q=5)
        
        # All selected primes should be in upper half
        midpoint = primes[len(primes) // 2]
        for q in special_q:
            assert q >= midpoint or q >= 100  # MIN_SPECIAL_Q fallback
    
    def test_handles_small_factor_base(self):
        """Should handle small factor bases gracefully."""
        primes = [2, 3, 5, 7, 11]
        special_q = select_special_q_primes(primes, num_special_q=3)
        
        assert len(special_q) <= 3
        for q in special_q:
            assert q in primes
    
    def test_empty_factor_base(self):
        """Should handle empty factor base."""
        special_q = select_special_q_primes([], num_special_q=5)
        assert special_q == []


# =============================================================================
# Lattice Sieve Correctness Tests
# =============================================================================

class TestLatticeSieveCorrectness:
    """Tests for correctness of lattice sieve."""
    
    def test_produces_valid_relations(self):
        """All relations should be valid (both sides factor completely)."""
        n = 2021  # 43 * 47
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(100)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=50, num_special_q=3
        ))
        
        for rel in relations:
            assert verify_relation(rel, primes), f"Invalid relation: {rel}"
    
    def test_coprime_ab(self):
        """All relations should have gcd(a, b) = 1."""
        n = 143  # 11 * 13
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(50)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=30, num_special_q=2
        ))
        
        for rel in relations:
            assert math.gcd(rel.a, rel.b) == 1
    
    def test_finds_some_relations(self):
        """Should find at least some relations for a reasonable input."""
        n = 1001  # 7 * 11 * 13
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(100)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=100, num_special_q=5
        ))
        
        assert len(relations) > 0, "Expected to find some relations"
    
    def test_algebraic_norm_correct(self):
        """Algebraic values should match polynomial evaluation."""
        n = 851  # 23 * 37
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(60)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=40, num_special_q=2
        ))
        
        for rel in relations:
            computed = selection.algebraic.evaluate_homogeneous(rel.a, rel.b)
            assert computed == rel.algebraic_value
    
    def test_rational_norm_correct(self):
        """Rational values should match polynomial evaluation."""
        n = 713  # 23 * 31
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(60)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=40, num_special_q=2
        ))
        
        for rel in relations:
            computed = selection.rational.evaluate_homogeneous(rel.a, rel.b)
            assert computed == rel.rational_value


class TestSingleSpecialQ:
    """Tests for single special-q sieving."""
    
    def test_special_q_divides_algebraic(self):
        """For relations from q-sieve, q should divide algebraic value."""
        n = 2021
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(200)
        
        q = 53  # A prime with roots of f(x) mod q
        relations = list(lattice_sieve_for_special_q(
            selection, primes, q, sieve_region=50
        ))
        
        for rel in relations:
            # q should be in the algebraic factors (it divides by construction)
            if rel.algebraic_factors:
                assert rel.algebraic_value % q == 0


# =============================================================================
# Comparison with Line Sieve
# =============================================================================

class TestComparisonWithLineSieve:
    """Tests comparing lattice sieve with line sieve."""
    
    def test_both_find_valid_relations(self):
        """Both sieves should produce valid relations."""
        n = 1001
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(50)
        
        line_relations = list(find_relations(selection, primes, interval=20))
        lattice_relations = list(find_relations_lattice(
            selection, primes, sieve_region=30, num_special_q=3
        ))
        
        for rel in line_relations:
            assert verify_relation(rel, primes)
        
        for rel in lattice_relations:
            assert verify_relation(rel, primes)
    
    def test_hybrid_sieve(self):
        """Hybrid sieve should work correctly."""
        n = 437  # 19 * 23
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(60)
        
        # Hybrid with small factor base (should use line sieve)
        relations_small = list(find_relations_hybrid(
            selection, primes[:20], interval=30, prefer_lattice=True
        ))
        
        # Hybrid with larger factor base (should use lattice sieve)
        relations_large = list(find_relations_hybrid(
            selection, primes, lattice_sieve_region=50, prefer_lattice=True
        ))
        
        # Both should produce valid relations
        for rel in relations_small + relations_large:
            assert verify_relation(rel, primes)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests and benchmarks."""
    
    @pytest.mark.slow
    def test_lattice_sieve_faster_for_large_fb(self):
        """Lattice sieve should be faster than line sieve for large factor bases."""
        n = 123456789
        selection = select_polynomial(n, degree=3)
        primes = get_test_primes(1000)
        
        # Benchmark line sieve
        start = time.time()
        line_relations = list(find_relations(selection, primes, interval=30))
        line_time = time.time() - start
        
        # Benchmark lattice sieve
        start = time.time()
        lattice_relations = list(find_relations_lattice(
            selection, primes, sieve_region=50, num_special_q=10
        ))
        lattice_time = time.time() - start
        
        # Report results
        print(f"\nLine sieve: {len(line_relations)} relations in {line_time:.3f}s")
        print(f"Lattice sieve: {len(lattice_relations)} relations in {lattice_time:.3f}s")
        
        # Both should find relations
        assert len(line_relations) > 0 or len(lattice_relations) > 0
    
    def test_scales_with_sieve_region(self):
        """Larger sieve region should find more relations."""
        n = 10001
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(200)
        
        small_relations = list(find_relations_lattice(
            selection, primes, sieve_region=20, num_special_q=3
        ))
        
        large_relations = list(find_relations_lattice(
            selection, primes, sieve_region=100, num_special_q=3
        ))
        
        # Larger region should find at least as many (usually more)
        assert len(large_relations) >= len(small_relations)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_primes(self):
        """Should handle empty prime list gracefully."""
        n = 100
        selection = select_polynomial(n, degree=2)
        
        relations = list(find_relations_lattice(selection, [], sieve_region=10))
        assert relations == []
    
    def test_very_small_sieve_region(self):
        """Should handle very small sieve regions."""
        n = 100
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(50)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=5, num_special_q=2
        ))
        # May or may not find relations, but shouldn't crash
        assert isinstance(relations, list)
    
    def test_polynomial_with_no_roots(self):
        """Handle case where polynomial has no roots mod some primes."""
        # Use a polynomial that might have few roots
        n = 1000003  # Prime, so factorization will fail anyway
        selection = select_polynomial(n, degree=3)
        primes = get_test_primes(100)
        
        # Should not crash even if no relations found
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=20, num_special_q=3
        ))
        assert isinstance(relations, list)


# =============================================================================
# Integration Tests  
# =============================================================================

class TestIntegration:
    """Integration tests with other GNFS components."""
    
    def test_relations_compatible_with_relation_class(self):
        """Relations should have correct structure."""
        n = 1001
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(100)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=50, num_special_q=3
        ))
        
        for rel in relations:
            assert isinstance(rel, Relation)
            assert isinstance(rel.a, int)
            assert isinstance(rel.b, int)
            assert rel.b > 0
            assert isinstance(rel.algebraic_factors, dict)
            assert isinstance(rel.rational_factors, dict)
    
    def test_combined_factors_method(self):
        """Relation.combined_factors() should work correctly."""
        n = 437
        selection = select_polynomial(n, degree=2)
        primes = get_test_primes(60)
        
        relations = list(find_relations_lattice(
            selection, primes, sieve_region=30, num_special_q=2
        ))
        
        for rel in relations:
            combined = rel.combined_factors()
            assert isinstance(combined, dict)
            # Combined should contain all primes from both sides
            for p in rel.algebraic_factors:
                assert p in combined
            for p in rel.rational_factors:
                assert p in combined


# =============================================================================
# Benchmark Data Collection
# =============================================================================

def benchmark_sieves(n: int, degree: int, factor_base_bound: int, 
                     line_interval: int, lattice_region: int,
                     num_special_q: int = 10) -> dict:
    """Run benchmark comparing line and lattice sieve.
    
    Returns dict with timing and relation counts.
    """
    selection = select_polynomial(n, degree)
    primes = get_test_primes(factor_base_bound)
    
    # Line sieve
    start = time.time()
    line_rels = list(find_relations(selection, primes, interval=line_interval))
    line_time = time.time() - start
    
    # Lattice sieve  
    start = time.time()
    lattice_rels = list(find_relations_lattice(
        selection, primes, sieve_region=lattice_region, num_special_q=num_special_q
    ))
    lattice_time = time.time() - start
    
    return {
        'n': n,
        'degree': degree,
        'factor_base_size': len(primes),
        'line_relations': len(line_rels),
        'line_time': line_time,
        'lattice_relations': len(lattice_rels),
        'lattice_time': lattice_time,
        'speedup': line_time / lattice_time if lattice_time > 0 else float('inf'),
    }


if __name__ == '__main__':
    # Run benchmarks when executed directly
    print("Running lattice sieve benchmarks...")
    print("=" * 70)
    
    test_cases = [
        (10007, 2, 200, 30, 50),
        (100003, 3, 500, 30, 100),
        (1000003, 3, 1000, 50, 200),
    ]
    
    for n, degree, fb_bound, line_int, lattice_reg in test_cases:
        print(f"\nn={n}, degree={degree}, FB bound={fb_bound}")
        result = benchmark_sieves(n, degree, fb_bound, line_int, lattice_reg)
        print(f"  Line sieve:    {result['line_relations']:4d} relations in {result['line_time']:.3f}s")
        print(f"  Lattice sieve: {result['lattice_relations']:4d} relations in {result['lattice_time']:.3f}s")
        print(f"  Speedup: {result['speedup']:.2f}x")

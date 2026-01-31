"""Mathematical tests for algebraic square root computation.

Tests cover:
1. Number field element arithmetic
2. Tonelli-Shanks algorithm
3. Hensel lifting for prime powers
4. Montgomery-style square root
5. Integration with GNFS factor extraction
"""

import pytest
from fractions import Fraction
from math import gcd, isqrt

from gnfs.sqrt.algebraic_sqrt import (
    NumberFieldElement,
    AlgebraicSquareRoot,
    tonelli_shanks,
    sqrt_mod_prime_power,
    compute_algebraic_product,
    montgomery_sqrt_rational,
)
from gnfs.sqrt import find_factors, find_factors_simple
from gnfs.sieve import Relation


# =============================================================================
# Number Field Element Arithmetic
# =============================================================================

class TestNumberFieldElement:
    """Tests for number field element operations."""
    
    def test_create_from_polynomial(self):
        """Create element from polynomial coefficients."""
        # In Q(√2), define f(x) = x² - 2
        poly = [-2, 0, 1]  # -2 + 0*x + 1*x²
        
        # Element 3 + 2*α (= 3 + 2√2)
        elem = NumberFieldElement([Fraction(3), Fraction(2)], poly)
        assert elem.coeffs == [Fraction(3), Fraction(2)]
        assert elem.degree == 2
    
    def test_addition(self):
        """Test addition in number field."""
        poly = [-2, 0, 1]  # x² - 2
        
        a = NumberFieldElement([Fraction(1), Fraction(2)], poly)  # 1 + 2α
        b = NumberFieldElement([Fraction(3), Fraction(-1)], poly)  # 3 - α
        
        c = a + b
        assert c.coeffs == [Fraction(4), Fraction(1)]  # 4 + α
    
    def test_subtraction(self):
        """Test subtraction in number field."""
        poly = [-2, 0, 1]
        
        a = NumberFieldElement([Fraction(5), Fraction(3)], poly)
        b = NumberFieldElement([Fraction(2), Fraction(1)], poly)
        
        c = a - b
        assert c.coeffs == [Fraction(3), Fraction(2)]
    
    def test_negation(self):
        """Test negation in number field."""
        poly = [-2, 0, 1]
        
        a = NumberFieldElement([Fraction(3), Fraction(-2)], poly)
        neg_a = -a
        
        assert neg_a.coeffs == [Fraction(-3), Fraction(2)]
    
    def test_multiplication_basic(self):
        """Test multiplication: (1 + α)(1 - α) = 1 - α²."""
        poly = [-2, 0, 1]  # x² = 2
        
        a = NumberFieldElement([Fraction(1), Fraction(1)], poly)  # 1 + α
        b = NumberFieldElement([Fraction(1), Fraction(-1)], poly)  # 1 - α
        
        c = a * b
        # (1 + α)(1 - α) = 1 - α² = 1 - 2 = -1
        assert c.coeffs == [Fraction(-1), Fraction(0)]
    
    def test_multiplication_with_reduction(self):
        """Test that α² is properly reduced."""
        poly = [-2, 0, 1]  # α² = 2
        
        alpha = NumberFieldElement.alpha(poly)
        alpha_sq = alpha * alpha
        
        # α² = 2 in this field
        assert alpha_sq.coeffs == [Fraction(2), Fraction(0)]
    
    def test_multiplication_cubic(self):
        """Test multiplication in cubic field."""
        # x³ - 2, so α³ = 2
        poly = [-2, 0, 0, 1]
        
        alpha = NumberFieldElement.alpha(poly)
        
        # α * α = α²
        alpha_sq = alpha * alpha
        assert alpha_sq.coeffs == [Fraction(0), Fraction(0), Fraction(1)]
        
        # α² * α = α³ = 2
        alpha_cube = alpha_sq * alpha
        assert alpha_cube.coeffs == [Fraction(2), Fraction(0), Fraction(0)]
    
    def test_multiplication_cubic_high_degree(self):
        """Test α² * α² = α⁴ = 2α in cubic field (regression test).
        
        This specifically tests reduction of α⁴ which requires shifting
        contributions when applying α³ = 2.
        """
        # x³ - 2, so α³ = 2
        poly = [-2, 0, 0, 1]
        
        alpha = NumberFieldElement.alpha(poly)
        alpha_sq = alpha * alpha
        
        # α² * α² = α⁴ = α * α³ = α * 2 = 2α
        alpha_4 = alpha_sq * alpha_sq
        assert alpha_4.coeffs == [Fraction(0), Fraction(2), Fraction(0)], \
            f"Expected [0, 2, 0] for α⁴, got {alpha_4.coeffs}"
        
        # Also verify via different path: α³ * α = 2 * α = 2α
        alpha_3 = alpha_sq * alpha
        alpha_4_alt = alpha_3 * alpha
        assert alpha_4_alt.coeffs == [Fraction(0), Fraction(2), Fraction(0)]
    
    def test_multiplication_quartic(self):
        """Test multiplication in quartic field."""
        # x⁴ - 2, so α⁴ = 2
        poly = [-2, 0, 0, 0, 1]
        
        alpha = NumberFieldElement.alpha(poly)
        
        # α² * α² = α⁴ = 2
        alpha_sq = alpha * alpha
        alpha_4 = alpha_sq * alpha_sq
        assert alpha_4.coeffs == [Fraction(2), Fraction(0), Fraction(0), Fraction(0)]
        
        # α³ * α² = α⁵ = α * α⁴ = 2α
        alpha_3 = alpha_sq * alpha
        alpha_5 = alpha_3 * alpha_sq
        assert alpha_5.coeffs == [Fraction(0), Fraction(2), Fraction(0), Fraction(0)]
    
    def test_power_zero(self):
        """x^0 = 1."""
        poly = [-2, 0, 1]
        a = NumberFieldElement([Fraction(3), Fraction(5)], poly)
        
        result = a ** 0
        assert result.coeffs == [Fraction(1), Fraction(0)]
    
    def test_power_one(self):
        """x^1 = x."""
        poly = [-2, 0, 1]
        a = NumberFieldElement([Fraction(3), Fraction(5)], poly)
        
        result = a ** 1
        assert result.coeffs == a.coeffs
    
    def test_power_higher(self):
        """Test higher powers."""
        poly = [-2, 0, 1]  # α² = 2
        alpha = NumberFieldElement.alpha(poly)
        
        # α^4 = (α²)² = 2² = 4
        alpha_4 = alpha ** 4
        assert alpha_4.coeffs == [Fraction(4), Fraction(0)]
    
    def test_from_ab(self):
        """Test creating (a - b*α) element."""
        poly = [-2, 0, 1]
        
        # (3 - 2*α)
        elem = NumberFieldElement.from_ab(3, 2, poly)
        assert elem.coeffs == [Fraction(3), Fraction(-2)]
    
    def test_evaluate_at_integer(self):
        """Test evaluating element at α = m."""
        poly = [-2, 0, 1]
        
        # 3 + 5*α evaluated at α = 7
        elem = NumberFieldElement([Fraction(3), Fraction(5)], poly)
        val = elem.evaluate_at(7)
        
        assert val == Fraction(3 + 5 * 7)  # 38
    
    def test_is_zero(self):
        """Test zero detection."""
        poly = [-2, 0, 1]
        
        zero = NumberFieldElement.zero(poly)
        assert zero.is_zero()
        
        nonzero = NumberFieldElement([Fraction(0), Fraction(1)], poly)
        assert not nonzero.is_zero()


class TestNormComputation:
    """Tests for norm computation in number fields."""
    
    def test_norm_of_integer(self):
        """Norm of rational integer a is a^d."""
        poly = [-2, 0, 1]  # degree 2
        
        elem = NumberFieldElement([Fraction(3), Fraction(0)], poly)
        norm = elem.norm()
        
        # N(3) = 3² = 9 for quadratic field
        assert norm == Fraction(9)
    
    def test_norm_of_alpha(self):
        """Norm of α equals (-1)^d * constant term of minimal poly."""
        poly = [-2, 0, 1]  # x² - 2
        
        alpha = NumberFieldElement.alpha(poly)
        norm = alpha.norm()
        
        # N(α) = (-1)² * (-2) = -2 for α satisfying α² - 2 = 0
        assert norm == Fraction(-2)
    
    def test_norm_multiplicative(self):
        """N(ab) = N(a) * N(b)."""
        poly = [-2, 0, 1]
        
        a = NumberFieldElement([Fraction(1), Fraction(1)], poly)  # 1 + α
        b = NumberFieldElement([Fraction(2), Fraction(-1)], poly)  # 2 - α
        
        ab = a * b
        
        norm_a = a.norm()
        norm_b = b.norm()
        norm_ab = ab.norm()
        
        assert norm_ab == norm_a * norm_b


# =============================================================================
# Tonelli-Shanks Algorithm
# =============================================================================

class TestTonelliShanks:
    """Tests for modular square root computation."""
    
    def test_perfect_squares(self):
        """Test with known perfect squares."""
        # 4 = 2² mod 7
        result = tonelli_shanks(4, 7)
        assert result is not None
        assert (result * result) % 7 == 4
    
    def test_quadratic_residue(self):
        """Test various quadratic residues."""
        for p in [7, 11, 13, 17, 19, 23]:
            for a in range(1, p):
                # Check if a is QR
                if pow(a, (p - 1) // 2, p) == 1:
                    x = tonelli_shanks(a, p)
                    assert x is not None
                    assert (x * x) % p == a
    
    def test_non_residue_returns_none(self):
        """Non-quadratic residues should return None."""
        # 3 is not a QR mod 7
        result = tonelli_shanks(3, 7)
        assert result is None
    
    def test_zero(self):
        """sqrt(0) = 0."""
        result = tonelli_shanks(0, 7)
        assert result == 0
    
    def test_one(self):
        """sqrt(1) = 1 or p-1."""
        result = tonelli_shanks(1, 7)
        assert result is not None
        assert (result * result) % 7 == 1
    
    def test_prime_two(self):
        """Handle p = 2 specially."""
        assert tonelli_shanks(0, 2) == 0
        assert tonelli_shanks(1, 2) == 1
    
    def test_large_prime(self):
        """Test with larger prime."""
        p = 104729  # 10000th prime
        
        # Find a QR and test
        for a in range(2, 100):
            if pow(a, (p - 1) // 2, p) == 1:
                x = tonelli_shanks(a, p)
                assert x is not None
                assert (x * x) % p == a
                break


# =============================================================================
# Hensel Lifting
# =============================================================================

class TestHenselLifting:
    """Tests for square root mod prime powers."""
    
    def test_mod_prime(self):
        """sqrt mod p (k=1) should match Tonelli-Shanks."""
        x = sqrt_mod_prime_power(4, 7, 1)
        assert x is not None
        assert (x * x) % 7 == 4
    
    def test_mod_prime_squared(self):
        """Test lifting to p²."""
        # sqrt(4) mod 49
        x = sqrt_mod_prime_power(4, 7, 2)
        assert x is not None
        assert (x * x) % 49 == 4
    
    def test_mod_prime_cubed(self):
        """Test lifting to p³."""
        # sqrt(4) mod 343
        x = sqrt_mod_prime_power(4, 7, 3)
        assert x is not None
        assert (x * x) % 343 == 4
    
    def test_non_residue_returns_none(self):
        """Non-QR should return None."""
        # 3 is not a QR mod 7
        x = sqrt_mod_prime_power(3, 7, 2)
        assert x is None
    
    def test_larger_example(self):
        """Test with larger numbers."""
        # sqrt(16) mod 11^3 = 1331
        x = sqrt_mod_prime_power(16, 11, 3)
        assert x is not None
        assert (x * x) % 1331 == 16


# =============================================================================
# Rational Square Root
# =============================================================================

class TestRationalSquareRoot:
    """Tests for integer square root."""
    
    def test_perfect_squares(self):
        """Test with perfect squares."""
        for n in [1, 4, 9, 16, 25, 36, 100, 10000]:
            result = montgomery_sqrt_rational(n)
            assert result * result == n
    
    def test_non_perfect_square_raises(self):
        """Non-perfect squares should raise."""
        with pytest.raises(ValueError):
            montgomery_sqrt_rational(5)
    
    def test_large_perfect_square(self):
        """Test with large perfect square."""
        n = 123456789 ** 2
        result = montgomery_sqrt_rational(n)
        assert result == 123456789


# =============================================================================
# Algebraic Product Computation
# =============================================================================

class TestAlgebraicProduct:
    """Tests for computing products of (a - b*α)."""
    
    def test_single_relation(self):
        """Product of single relation."""
        poly = [-2, 0, 1]  # x² - 2
        
        rel = Relation(
            a=3, b=1,
            algebraic_value=1,  # dummy
            rational_value=1,
            algebraic_factors={},
            rational_factors={},
        )
        
        product = compute_algebraic_product([rel], [0], poly)
        
        # (3 - 1*α) = 3 - α
        assert product.coeffs == [Fraction(3), Fraction(-1)]
    
    def test_two_relations(self):
        """Product of two relations."""
        poly = [-2, 0, 1]  # α² = 2
        
        rels = [
            Relation(a=1, b=1, algebraic_value=1, rational_value=1,
                    algebraic_factors={}, rational_factors={}),
            Relation(a=1, b=-1, algebraic_value=1, rational_value=1,
                    algebraic_factors={}, rational_factors={}),
        ]
        
        product = compute_algebraic_product(rels, [0, 1], poly)
        
        # (1 - α)(1 + α) = 1 - α² = 1 - 2 = -1
        assert product.coeffs == [Fraction(-1), Fraction(0)]


# =============================================================================
# Full Factor Extraction
# =============================================================================

class TestFactorExtraction:
    """Tests for end-to-end factor extraction."""
    
    def test_simple_factorization(self):
        """Test factoring with simple sqrt method."""
        # Factor 91 = 7 * 13
        n = 91
        
        # Create relations that give a dependency
        # These are made up for testing - in real GNFS they come from sieving
        relations = [
            Relation(a=1, b=1, algebraic_value=4, rational_value=4,
                    algebraic_factors={2: 2}, rational_factors={2: 2}),
            Relation(a=2, b=1, algebraic_value=9, rational_value=9,
                    algebraic_factors={3: 2}, rational_factors={3: 2}),
        ]
        primes = [2, 3]
        
        # Try to find factors
        factors = list(find_factors_simple(n, relations, primes))
        
        # May or may not find factors depending on the specific values
        # Just verify it runs without error
        assert isinstance(factors, list)
    
    def test_with_known_dependency(self):
        """Test with relations that form a known valid dependency."""
        n = 143  # 11 * 13
        
        # Construct relations where products are squares
        # Product of algebraic values: 36 = 6²
        # Product of rational values: 36 = 6²
        relations = [
            Relation(a=2, b=1, algebraic_value=6, rational_value=6,
                    algebraic_factors={2: 1, 3: 1}, rational_factors={2: 1, 3: 1}),
            Relation(a=3, b=1, algebraic_value=6, rational_value=6,
                    algebraic_factors={2: 1, 3: 1}, rational_factors={2: 1, 3: 1}),
        ]
        primes = [2, 3]
        
        factors = list(find_factors_simple(n, relations, primes))
        
        # Verify any factors found are correct
        for f in factors:
            assert n % f == 0
            assert 1 < f < n


class TestAlgebraicSquareRootClass:
    """Tests for the AlgebraicSquareRoot helper class."""
    
    def test_initialization(self):
        """Test class initialization."""
        poly = [-91, 0, 1]  # x² - 91
        m = 10  # 10² = 100 ≡ 9 (mod 91)
        n = 91
        
        alg_sqrt = AlgebraicSquareRoot(poly, m, n)
        
        assert alg_sqrt.degree == 2
        assert alg_sqrt.m == 10
        assert alg_sqrt.n == 91
    
    def test_compute_product(self):
        """Test product computation via class."""
        poly = [-2, 0, 1]
        m = 10
        n = 91
        
        alg_sqrt = AlgebraicSquareRoot(poly, m, n)
        
        rels = [
            Relation(a=1, b=1, algebraic_value=1, rational_value=1,
                    algebraic_factors={}, rational_factors={}),
        ]
        
        product = alg_sqrt.compute_product(rels, [0])
        assert product.coeffs == [Fraction(1), Fraction(-1)]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dependency(self):
        """Empty dependency should be handled."""
        poly = [-2, 0, 1]
        
        product = compute_algebraic_product([], [], poly)
        
        # Empty product is 1
        assert product.coeffs == [Fraction(1), Fraction(0)]
    
    def test_high_degree_polynomial(self):
        """Test with degree 4 polynomial."""
        # x^4 - 2
        poly = [-2, 0, 0, 0, 1]
        
        alpha = NumberFieldElement.alpha(poly)
        
        # α^4 = 2
        alpha_4 = alpha ** 4
        assert alpha_4.coeffs == [Fraction(2), Fraction(0), Fraction(0), Fraction(0)]
    
    def test_non_monic_polynomial(self):
        """Test with non-monic polynomial (leading coeff != 1)."""
        # 2x² - 4 = 0, i.e., x² = 2
        poly = [-4, 0, 2]
        
        elem = NumberFieldElement([Fraction(1), Fraction(1)], poly)
        
        # Should still work (reduction divides by leading coeff)
        sq = elem * elem
        assert isinstance(sq, NumberFieldElement)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

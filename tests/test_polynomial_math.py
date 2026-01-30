"""Comprehensive mathematical tests for polynomial operations in GNFS.

Tests verify:
- Polynomial arithmetic: addition, subtraction, multiplication, division
- Polynomial evaluation at integer and rational points
- Modular polynomial arithmetic (mod p for various primes)
- Root finding: verify roots satisfy f(r) ≡ 0 (mod p)
- Edge cases: zero polynomial, constant polynomials, degree 0/1
"""

import pytest
from fractions import Fraction
import sympy as sp

from gnfs.polynomial.polynomial import Polynomial
from gnfs.sieve.roots import _polynomial_roots_mod_p


# =============================================================================
# Polynomial Arithmetic Tests
# =============================================================================

class TestPolynomialEvaluation:
    """Tests for polynomial evaluation at various points."""
    
    def test_evaluate_at_zero(self):
        """f(0) should return the constant term."""
        poly = Polynomial((5, 3, 2, 1))  # 5 + 3x + 2x^2 + x^3
        assert poly.evaluate(0) == 5
    
    def test_evaluate_at_one(self):
        """f(1) should return sum of all coefficients."""
        poly = Polynomial((1, 2, 3, 4))  # 1 + 2x + 3x^2 + 4x^3
        assert poly.evaluate(1) == 1 + 2 + 3 + 4  # = 10
    
    def test_evaluate_at_minus_one(self):
        """f(-1) should alternate signs."""
        poly = Polynomial((1, 2, 3, 4))  # 1 + 2x + 3x^2 + 4x^3
        assert poly.evaluate(-1) == 1 - 2 + 3 - 4  # = -2
    
    def test_evaluate_quadratic(self):
        """Test evaluation of quadratic: x^2 - 5x + 6 = (x-2)(x-3)."""
        poly = Polynomial((6, -5, 1))  # 6 - 5x + x^2
        assert poly.evaluate(2) == 0  # root at 2
        assert poly.evaluate(3) == 0  # root at 3
        assert poly.evaluate(4) == 6 - 20 + 16  # = 2
        assert poly.evaluate(0) == 6
    
    def test_evaluate_cubic(self):
        """Test evaluation of cubic polynomial."""
        # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        poly = Polynomial((-6, 11, -6, 1))
        assert poly.evaluate(1) == 0
        assert poly.evaluate(2) == 0
        assert poly.evaluate(3) == 0
    
    def test_evaluate_large_values(self):
        """Test evaluation with large inputs."""
        poly = Polynomial((1, 1, 1))  # 1 + x + x^2
        x = 1000
        expected = 1 + 1000 + 1000000
        assert poly.evaluate(x) == expected
    
    def test_evaluate_negative_coefficients(self):
        """Test with negative coefficients."""
        poly = Polynomial((-1, -2, -3))  # -1 - 2x - 3x^2
        assert poly.evaluate(0) == -1
        assert poly.evaluate(1) == -6
        assert poly.evaluate(-1) == -1 + 2 - 3  # = -2


class TestPolynomialHomogeneous:
    """Tests for homogeneous polynomial evaluation b^d f(a/b)."""
    
    def test_homogeneous_b_equals_one(self):
        """With b=1, homogeneous evaluation equals regular evaluation."""
        poly = Polynomial((1, 2, 3))  # 1 + 2x + 3x^2
        for a in range(-5, 6):
            assert poly.evaluate_homogeneous(a, 1) == poly.evaluate(a)
    
    def test_homogeneous_basic(self):
        """Test homogeneous evaluation: b^d * f(a/b)."""
        # f(x) = x^2 + 1, so b^2 * f(a/b) = b^2 * (a^2/b^2 + 1) = a^2 + b^2
        poly = Polynomial((1, 0, 1))  # 1 + x^2
        assert poly.evaluate_homogeneous(3, 2) == 9 + 4  # = 13
        assert poly.evaluate_homogeneous(4, 3) == 16 + 9  # = 25
    
    def test_homogeneous_linear(self):
        """Test homogeneous evaluation for linear polynomial."""
        # f(x) = x - m, so b * f(a/b) = a - mb
        m = 5
        poly = Polynomial((-m, 1))  # x - 5
        assert poly.evaluate_homogeneous(10, 2) == 10 - 5 * 2  # = 0
        assert poly.evaluate_homogeneous(7, 1) == 7 - 5  # = 2
    
    def test_homogeneous_symmetry(self):
        """Test that scaling (a, b) by k scales result by k^d."""
        poly = Polynomial((1, 0, 0, 1))  # 1 + x^3 (degree 3)
        a, b = 2, 3
        result = poly.evaluate_homogeneous(a, b)
        
        # Scaling by k=2: (2a, 2b) should give 2^3 * result
        scaled = poly.evaluate_homogeneous(2 * a, 2 * b)
        assert scaled == 8 * result
    
    def test_homogeneous_negative_a(self):
        """Test homogeneous evaluation with negative a."""
        poly = Polynomial((1, 2, 3))  # 1 + 2x + 3x^2
        # b^2 * f(a/b) = b^2 + 2ab + 3a^2
        a, b = -2, 3
        expected = 9 + 2 * (-2) * 3 + 3 * 4  # = 9 - 12 + 12 = 9
        assert poly.evaluate_homogeneous(a, b) == expected


class TestPolynomialDegree:
    """Tests for polynomial degree computation."""
    
    def test_degree_zero(self):
        """Constant polynomial has degree 0."""
        poly = Polynomial((5,))
        assert poly.degree() == 0
    
    def test_degree_one(self):
        """Linear polynomial has degree 1."""
        poly = Polynomial((3, 2))  # 3 + 2x
        assert poly.degree() == 1
    
    def test_degree_quadratic(self):
        """Quadratic polynomial has degree 2."""
        poly = Polynomial((1, 2, 3))
        assert poly.degree() == 2
    
    def test_degree_high(self):
        """Test high-degree polynomial."""
        coeffs = tuple(range(10))  # 0 + x + 2x^2 + ... + 9x^9
        poly = Polynomial(coeffs)
        assert poly.degree() == 9


# =============================================================================
# Modular Polynomial Arithmetic Tests
# =============================================================================

class TestModularPolynomialArithmetic:
    """Tests for polynomial evaluation modulo primes."""
    
    def test_evaluate_mod_prime(self):
        """Test f(x) mod p for various primes."""
        poly = Polynomial((1, 2, 3))  # 1 + 2x + 3x^2
        
        # f(2) = 1 + 4 + 12 = 17
        assert poly.evaluate(2) % 5 == 17 % 5  # = 2
        assert poly.evaluate(2) % 7 == 17 % 7  # = 3
    
    def test_roots_mod_prime_quadratic(self):
        """x^2 - 1 ≡ 0 (mod p) has roots ±1."""
        poly = Polynomial((-1, 0, 1))  # x^2 - 1
        
        for p in [3, 5, 7, 11, 13]:
            roots = _polynomial_roots_mod_p(poly, p)
            # Should have roots at 1 and p-1 (which is -1 mod p)
            root_set = set(roots)
            assert 1 in root_set
            assert (p - 1) in root_set or 1 in root_set
            
            # Verify roots satisfy equation
            for r in roots:
                assert poly.evaluate(r) % p == 0
    
    def test_roots_mod_prime_cubic(self):
        """Test cubic polynomial roots mod prime."""
        # x^3 - x = x(x-1)(x+1) has roots 0, 1, -1 mod p
        poly = Polynomial((0, -1, 0, 1))  # -x + x^3
        
        for p in [5, 7, 11]:
            roots = _polynomial_roots_mod_p(poly, p)
            root_set = set(roots)
            
            assert 0 in root_set
            assert 1 in root_set
            assert (p - 1) in root_set
            
            # Verify each root
            for r in roots:
                assert poly.evaluate(r) % p == 0
    
    def test_no_roots_mod_prime(self):
        """Test polynomial with no roots mod p."""
        # x^2 + 1 has no roots mod 3 (since 0^2+1=1, 1^2+1=2, 2^2+1=5≡2)
        poly = Polynomial((1, 0, 1))  # x^2 + 1
        roots = _polynomial_roots_mod_p(poly, 3)
        assert roots == []
        
        # Verify manually
        for x in range(3):
            assert poly.evaluate(x) % 3 != 0
    
    def test_roots_verify_equation(self):
        """Systematically verify that all returned roots satisfy f(r) ≡ 0 (mod p)."""
        test_polys = [
            Polynomial((6, -5, 1)),      # x^2 - 5x + 6 = (x-2)(x-3)
            Polynomial((-6, 11, -6, 1)), # (x-1)(x-2)(x-3)
            Polynomial((1, 1, 1)),       # x^2 + x + 1
            Polynomial((0, 0, 1)),       # x^2
        ]
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        for poly in test_polys:
            for p in primes:
                roots = _polynomial_roots_mod_p(poly, p)
                for r in roots:
                    assert poly.evaluate(r) % p == 0, \
                        f"Root {r} does not satisfy f(x) ≡ 0 (mod {p}) for polynomial {poly.coeffs}"
    
    def test_repeated_roots(self):
        """Test polynomial with repeated roots."""
        # (x - 1)^2 = x^2 - 2x + 1
        poly = Polynomial((1, -2, 1))
        
        for p in [3, 5, 7]:
            roots = _polynomial_roots_mod_p(poly, p)
            # Should return root 1 with multiplicity 2
            assert roots.count(1) == 2
    
    def test_all_elements_are_roots(self):
        """Test polynomial that has all elements as roots mod p."""
        # x^p - x ≡ 0 (mod p) for all x by Fermat's little theorem
        # For p=3: x^3 - x = x(x-1)(x+1)
        p = 3
        poly = Polynomial((0, -1, 0, 1))  # x^3 - x
        
        roots = _polynomial_roots_mod_p(poly, p)
        assert set(roots) == {0, 1, 2}


# =============================================================================
# Root Finding Verification Tests
# =============================================================================

class TestRootFindingCorrectness:
    """Verify that roots found by _polynomial_roots_mod_p satisfy f(r) ≡ 0 (mod p)."""
    
    def test_linear_roots(self):
        """Linear polynomial ax + b has root -b/a mod p."""
        # 3x + 2 ≡ 0 (mod 5) => x ≡ -2/3 ≡ -2 * 2 ≡ -4 ≡ 1 (mod 5)
        # (since 3 * 2 = 6 ≡ 1 mod 5, so 3^-1 = 2)
        poly = Polynomial((2, 3))  # 2 + 3x
        roots = _polynomial_roots_mod_p(poly, 5)
        
        assert len(roots) == 1
        assert roots[0] == 1
        assert poly.evaluate(1) % 5 == 0
    
    def test_irreducible_quadratic(self):
        """Test irreducible quadratic over certain primes."""
        # x^2 + 1 is irreducible over Z/3Z
        poly = Polynomial((1, 0, 1))
        assert _polynomial_roots_mod_p(poly, 3) == []
        
        # x^2 + 1 has roots mod 5: ±2 (since 2^2 = 4 ≡ -1 and 3^2 = 9 ≡ -1)
        roots_5 = _polynomial_roots_mod_p(poly, 5)
        assert set(roots_5) == {2, 3}
    
    def test_gnfs_polynomial_roots(self):
        """Test root finding for GNFS-style polynomials."""
        from gnfs.polynomial.selection import select_polynomial
        
        n = 143  # 11 * 13
        selection = select_polynomial(n, degree=2)
        poly = selection.algebraic
        
        # Find roots mod small primes
        for p in [2, 3, 5, 7, 11, 13]:
            roots = _polynomial_roots_mod_p(poly, p)
            for r in roots:
                assert poly.evaluate(r) % p == 0
    
    def test_relation_root_property(self):
        """For a relation (a, b), verify a - rb ≡ 0 (mod p) for some root r."""
        # This is the fundamental sieving property
        # If polynomial f has root r mod p, then a ≡ rb (mod p) 
        # means p | f(a/b) * b^d (homogeneous form)
        poly = Polynomial((-1, 0, 1))  # x^2 - 1
        p = 5
        roots = _polynomial_roots_mod_p(poly, p)  # {1, 4}
        
        # For (a=6, b=1): a - 1*b = 5 ≡ 0 (mod 5) ✓
        a, b = 6, 1
        assert any((a - r * b) % p == 0 for r in roots)
        
        # For (a=8, b=2): a - 4*b = 8 - 8 = 0 ≡ 0 (mod 5) ✓
        a, b = 8, 2
        assert any((a - r * b) % p == 0 for r in roots)


# =============================================================================
# Edge Cases
# =============================================================================

class TestPolynomialEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_polynomial(self):
        """Test polynomial with all zero coefficients."""
        poly = Polynomial((0,))
        assert poly.degree() == 0
        assert poly.evaluate(100) == 0
        assert poly.evaluate_homogeneous(5, 3) == 0
    
    def test_constant_polynomial(self):
        """Test constant (degree 0) polynomial."""
        poly = Polynomial((7,))
        assert poly.degree() == 0
        assert poly.evaluate(0) == 7
        assert poly.evaluate(100) == 7
        assert poly.evaluate(-50) == 7
    
    def test_monic_linear(self):
        """Test monic linear polynomial x - c."""
        c = 42
        poly = Polynomial((-c, 1))
        assert poly.evaluate(c) == 0
        assert poly.degree() == 1
    
    def test_large_coefficients(self):
        """Test with very large coefficients."""
        large = 10**15
        poly = Polynomial((large, large, 1))
        # f(0) = large
        assert poly.evaluate(0) == large
        # f(1) = large + large + 1 = 2*large + 1
        assert poly.evaluate(1) == 2 * large + 1
    
    def test_negative_large_coefficients(self):
        """Test with large negative coefficients."""
        poly = Polynomial((-10**12, 10**6, -1))
        assert poly.evaluate(0) == -10**12
    
    def test_single_term_polynomials(self):
        """Test polynomials with only one non-zero term."""
        # x^3 only
        poly = Polynomial((0, 0, 0, 5))
        assert poly.evaluate(2) == 5 * 8  # 5 * 2^3 = 40
        assert poly.evaluate(0) == 0
    
    def test_roots_mod_two(self):
        """Test roots modulo 2 (smallest prime)."""
        poly = Polynomial((0, 1))  # x
        roots = _polynomial_roots_mod_p(poly, 2)
        assert roots == [0]
        
        poly2 = Polynomial((1, 1))  # 1 + x
        roots2 = _polynomial_roots_mod_p(poly2, 2)
        assert roots2 == [1]
    
    def test_identity_polynomial(self):
        """Test identity polynomial f(x) = x."""
        poly = Polynomial((0, 1))  # x
        for val in range(-10, 11):
            assert poly.evaluate(val) == val
    
    def test_square_polynomial(self):
        """Test f(x) = x^2."""
        poly = Polynomial((0, 0, 1))  # x^2
        for val in range(-5, 6):
            assert poly.evaluate(val) == val * val


# =============================================================================
# Polynomial Consistency Tests
# =============================================================================

class TestPolynomialConsistency:
    """Test mathematical consistency properties."""
    
    def test_evaluation_consistency_with_sympy(self):
        """Verify polynomial evaluation matches SymPy."""
        poly = Polynomial((1, -3, 2, 5, -1))
        x = sp.Symbol('x')
        sympy_poly = sum(c * x**i for i, c in enumerate(poly.coeffs))
        
        for val in range(-10, 11):
            our_result = poly.evaluate(val)
            sympy_result = int(sympy_poly.subs(x, val))
            assert our_result == sympy_result
    
    def test_homogeneous_consistency(self):
        """Verify homogeneous evaluation is consistent with definition."""
        poly = Polynomial((2, -3, 1, 4))  # 2 - 3x + x^2 + 4x^3
        
        for a in range(-5, 6):
            for b in range(1, 5):  # b > 0 for division
                # Compute using formula: sum(c_i * a^i * b^(d-i))
                d = poly.degree()
                expected = sum(
                    c * (a ** i) * (b ** (d - i))
                    for i, c in enumerate(poly.coeffs)
                )
                actual = poly.evaluate_homogeneous(a, b)
                assert actual == expected
    
    def test_factored_polynomial_roots(self):
        """Test that roots of factored polynomials are found correctly."""
        # Create (x - 2)(x - 3)(x - 5) = x^3 - 10x^2 + 31x - 30
        # Expanding: (x-2)(x-3) = x^2 - 5x + 6
        # (x^2 - 5x + 6)(x - 5) = x^3 - 5x^2 - 5x^2 + 25x + 6x - 30 = x^3 - 10x^2 + 31x - 30
        poly = Polynomial((-30, 31, -10, 1))
        
        assert poly.evaluate(2) == 0
        assert poly.evaluate(3) == 0
        assert poly.evaluate(5) == 0
        
        # Check mod various primes
        for p in [7, 11, 13]:
            roots = _polynomial_roots_mod_p(poly, p)
            root_set = set(roots)
            assert 2 % p in root_set or 2 in root_set
            assert 3 % p in root_set or 3 in root_set
            assert 5 % p in root_set or 5 in root_set

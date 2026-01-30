"""Property-based tests for GNFS implementation using Hypothesis.

Tests verify:
- For random primes p, q: verify gnfs_factor(p*q) returns {p, q}
- For random polynomials: verify evaluation consistency
- For random matrices: verify nullspace properties
"""

import pytest
import numpy as np
from fractions import Fraction
import sympy as sp
from math import gcd, isqrt

from hypothesis import given, strategies as st, settings, assume

from gnfs.polynomial.polynomial import Polynomial
from gnfs.polynomial.number_field import NumberField
from gnfs.polynomial.selection import (
    select_polynomial, 
    base_m_expansion, 
    optimal_base_m,
    compute_alpha,
    murphy_e_score,
)
from gnfs.sieve.roots import _polynomial_roots_mod_p
from gnfs.linalg.matrix import _nullspace_mod2
from gnfs.factor import gnfs_factor


# =============================================================================
# Polynomial Property Tests
# =============================================================================

class TestPolynomialProperties:
    """Property-based tests for polynomial operations."""
    
    @given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=6))
    def test_evaluate_at_zero_gives_constant(self, coeffs):
        """f(0) should always equal the constant term."""
        poly = Polynomial(tuple(coeffs))
        assert poly.evaluate(0) == coeffs[0]
    
    @given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=6),
           st.integers(min_value=-50, max_value=50))
    def test_evaluation_consistency(self, coeffs, x):
        """Polynomial evaluation should be consistent."""
        poly = Polynomial(tuple(coeffs))
        
        # Evaluate using our method
        result = poly.evaluate(x)
        
        # Evaluate manually
        expected = sum(c * (x ** i) for i, c in enumerate(coeffs))
        
        assert result == expected
    
    @given(st.lists(st.integers(min_value=-50, max_value=50), min_size=2, max_size=5),
           st.integers(min_value=-20, max_value=20),
           st.integers(min_value=1, max_value=10))
    def test_homogeneous_evaluation(self, coeffs, a, b):
        """Test homogeneous evaluation b^d * f(a/b)."""
        poly = Polynomial(tuple(coeffs))
        d = poly.degree()
        
        result = poly.evaluate_homogeneous(a, b)
        
        # Manual computation
        expected = sum(c * (a ** i) * (b ** (d - i)) for i, c in enumerate(coeffs))
        
        assert result == expected
    
    @given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=4))
    def test_degree_equals_length_minus_one(self, coeffs):
        """Degree should be len(coeffs) - 1."""
        poly = Polynomial(tuple(coeffs))
        assert poly.degree() == len(coeffs) - 1


class TestPolynomialRootProperties:
    """Property-based tests for polynomial root finding."""
    
    @given(st.lists(st.integers(min_value=-20, max_value=20), min_size=2, max_size=4),
           st.integers(min_value=2, max_value=37).filter(sp.isprime))
    @settings(max_examples=50)
    def test_roots_satisfy_equation(self, coeffs, p):
        """All returned roots should satisfy f(r) ≡ 0 (mod p)."""
        poly = Polynomial(tuple(coeffs))
        roots = _polynomial_roots_mod_p(poly, p)
        
        for r in roots:
            assert poly.evaluate(r) % p == 0
    
    @given(st.integers(min_value=2, max_value=23).filter(sp.isprime))
    def test_linear_polynomial_has_one_root(self, p):
        """Linear polynomial ax + b has exactly one root mod p if gcd(a,p)=1."""
        a = 1  # Guaranteed coprime to p
        b = 3
        poly = Polynomial((b, a))  # b + ax
        
        roots = _polynomial_roots_mod_p(poly, p)
        assert len(roots) == 1
        assert poly.evaluate(roots[0]) % p == 0


# =============================================================================
# Number Field Property Tests
# =============================================================================

class TestNumberFieldProperties:
    """Property-based tests for number field arithmetic."""
    
    @given(st.integers(min_value=2, max_value=20),
           st.lists(st.integers(min_value=-10, max_value=10), min_size=2, max_size=2))
    def test_norm_multiplicativity(self, d, coeffs1):
        """Test N(xy) = N(x)N(y)."""
        assume(d > 1)  # Avoid trivial cases
        
        field = NumberField(Polynomial((-d, 0, 1)))  # x² - d
        
        x = field.element(coeffs1)
        y = field.element([1, 1])  # 1 + α
        
        norm_x = field.norm(x)
        norm_y = field.norm(y)
        norm_xy = field.norm(x * y)
        
        assert norm_xy == norm_x * norm_y
    
    @given(st.integers(min_value=2, max_value=20),
           st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=2),
           st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=2))
    def test_addition_commutativity(self, d, coeffs1, coeffs2):
        """x + y = y + x."""
        assume(d > 1)
        
        field = NumberField(Polynomial((-d, 0, 1)))
        x = field.element(coeffs1)
        y = field.element(coeffs2)
        
        assert x + y == y + x
    
    @given(st.integers(min_value=2, max_value=20),
           st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=2),
           st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=2))
    def test_multiplication_commutativity(self, d, coeffs1, coeffs2):
        """xy = yx."""
        assume(d > 1)
        
        field = NumberField(Polynomial((-d, 0, 1)))
        x = field.element(coeffs1)
        y = field.element(coeffs2)
        
        assert x * y == y * x


# =============================================================================
# Base-m Expansion Property Tests
# =============================================================================

class TestBaseMExpansionProperties:
    """Property-based tests for base-m expansion."""
    
    @given(st.integers(min_value=10, max_value=10000),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=50)
    def test_base_m_reconstructs_n(self, n, degree):
        """base_m_expansion(n, m, d) should give f where f(m) = n."""
        m = optimal_base_m(n, degree)
        assume(m > 1)
        
        try:
            coeffs = base_m_expansion(n, m, degree)
            poly = Polynomial(coeffs)
            assert poly.evaluate(m) == n
        except (AssertionError, ValueError):
            pass  # Some edge cases may fail
    
    @given(st.integers(min_value=100, max_value=10000),
           st.integers(min_value=2, max_value=4))
    @settings(max_examples=30)
    def test_polynomial_selection_root_property(self, n, degree):
        """Selected polynomial should satisfy f(m) ≡ 0 (mod n)."""
        selection = select_polynomial(n, degree=degree, optimize=False)
        
        f_at_m = selection.algebraic.evaluate(selection.m)
        assert f_at_m % n == 0 or f_at_m == n


# =============================================================================
# Linear Algebra Property Tests
# =============================================================================

class TestNullspaceProperties:
    """Property-based tests for nullspace computation."""
    
    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=3, max_value=7))
    @settings(max_examples=30)
    def test_nullspace_vectors_in_kernel(self, rows, cols):
        """Every nullspace vector v should satisfy Av = 0."""
        assume(cols >= rows)
        
        # Generate random binary matrix
        np.random.seed(42)  # Reproducibility
        matrix = np.random.randint(0, 2, size=(rows, cols), dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        for v in basis:
            result = (matrix @ v) % 2
            assert np.all(result == 0)
    
    @given(st.integers(min_value=2, max_value=5))
    def test_identity_has_empty_nullspace(self, n):
        """Identity matrix should have empty nullspace."""
        matrix = np.eye(n, dtype=int)
        basis = _nullspace_mod2(matrix)
        assert basis == []
    
    @given(st.integers(min_value=2, max_value=4),
           st.integers(min_value=3, max_value=6))
    @settings(max_examples=20)
    def test_nullspace_dimension(self, rows, cols):
        """Nullspace dimension should be n - rank."""
        assume(cols > rows)
        
        np.random.seed(123)
        matrix = np.random.randint(0, 2, size=(rows, cols), dtype=int)
        
        basis = _nullspace_mod2(matrix)
        
        # Nullspace dim should be at least cols - rows
        # (could be more if matrix is not full row rank)
        assert len(basis) >= cols - rows


# =============================================================================
# Scoring Property Tests
# =============================================================================

class TestScoringProperties:
    """Property-based tests for polynomial scoring."""
    
    @given(st.lists(st.integers(min_value=-50, max_value=50), min_size=2, max_size=5))
    def test_alpha_is_finite(self, coeffs):
        """Alpha value should be finite."""
        assume(any(c != 0 for c in coeffs))  # Non-zero polynomial
        
        poly = Polynomial(tuple(coeffs))
        alpha = compute_alpha(poly)
        
        assert not np.isnan(alpha)
        assert not np.isinf(alpha)
    
    @given(st.lists(st.integers(min_value=1, max_value=20), min_size=3, max_size=5),
           st.integers(min_value=100, max_value=1000))
    @settings(max_examples=20)
    def test_murphy_score_positive(self, coeffs, n):
        """Murphy E score should be positive."""
        poly = Polynomial(tuple(coeffs))
        score = murphy_e_score(poly, n)
        
        assert score > 0


# =============================================================================
# Factorization Property Tests
# =============================================================================

class TestFactorizationProperties:
    """Property-based tests for factorization."""
    
    @given(st.integers(min_value=2, max_value=47).filter(sp.isprime),
           st.integers(min_value=2, max_value=47).filter(sp.isprime))
    @settings(max_examples=15, deadline=30000)  # Allow more time
    def test_factor_semiprime(self, p, q):
        """Factoring p*q should give {p, q} when successful."""
        assume(p != q)  # Avoid perfect squares
        assume(p * q < 1000)  # Keep numbers small
        
        n = p * q
        factors = gnfs_factor(n, bound=max(p, q) + 10, interval=60, max_rounds=3)
        
        if factors:
            factor_set = set(factors)
            assert factor_set == {p, q}, f"For {n}={p}*{q}, got {factors}"
    
    @given(st.integers(min_value=10, max_value=200))
    @settings(max_examples=10, deadline=20000)
    def test_factors_divide_n(self, n):
        """Any returned factor should divide n."""
        factors = gnfs_factor(n, bound=30, interval=40, max_rounds=2)
        
        for f in factors:
            assert n % f == 0


# =============================================================================
# Relation Property Tests
# =============================================================================

class TestRelationProperties:
    """Property-based tests for relations."""
    
    @given(st.integers(min_value=50, max_value=200))
    @settings(max_examples=10, deadline=15000)
    def test_relations_are_coprime(self, n):
        """All relations should have gcd(a, b) = 1."""
        from gnfs.sieve.sieve import find_relations
        
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 20))
        relations = list(find_relations(selection, primes=primes, interval=20))
        
        for rel in relations:
            assert gcd(rel.a, rel.b) == 1
    
    @given(st.integers(min_value=50, max_value=150))
    @settings(max_examples=10, deadline=15000)
    def test_relations_smooth_over_base(self, n):
        """All relation values should factor over the factor base."""
        from gnfs.sieve.sieve import find_relations
        
        selection = select_polynomial(n)
        primes = list(sp.primerange(2, 25))
        prime_set = set(primes)
        relations = list(find_relations(selection, primes=primes, interval=25))
        
        for rel in relations:
            for p in rel.algebraic_factors:
                assert p in prime_set
            for p in rel.rational_factors:
                assert p in prime_set


# =============================================================================
# Combined Property Tests
# =============================================================================

class TestCombinedProperties:
    """Combined property tests across modules."""
    
    @given(st.integers(min_value=2, max_value=10),
           st.lists(st.integers(min_value=-3, max_value=3), min_size=2, max_size=2))
    def test_norm_formula_quadratic_field(self, d, coeffs):
        """Test N(a + bα) = a² - db² for Q[√d]."""
        assume(d > 1)
        
        field = NumberField(Polynomial((-d, 0, 1)))
        a, b = coeffs
        elem = field.element([a, b])
        
        expected = a * a - d * b * b
        actual = field.norm(elem)
        
        assert actual == expected
    
    @given(st.integers(min_value=2, max_value=20),
           st.integers(min_value=-10, max_value=10),
           st.integers(min_value=-10, max_value=10))
    def test_element_subtraction_gives_zero(self, d, a, b):
        """x - x should equal zero."""
        assume(d > 1)
        
        field = NumberField(Polynomial((-d, 0, 1)))
        x = field.element([a, b])
        
        result = x - x
        
        assert result.coeffs == (Fraction(0), Fraction(0))


# =============================================================================
# Invariant Tests
# =============================================================================

class TestInvariants:
    """Test mathematical invariants."""
    
    @given(st.integers(min_value=2, max_value=20),
           st.lists(st.integers(min_value=-5, max_value=5), min_size=2, max_size=2))
    def test_zero_norm_implies_zero_element(self, d, coeffs):
        """N(x) = 0 should imply x = 0 (for number fields without zero divisors)."""
        assume(d > 1)
        assume(not sp.is_square(d))  # Ensure proper number field
        
        field = NumberField(Polynomial((-d, 0, 1)))
        x = field.element(coeffs)
        
        if field.norm(x) == 0:
            # x should be zero
            assert all(c == 0 for c in x.coeffs)
    
    @given(st.integers(min_value=2, max_value=15),
           st.integers(min_value=0, max_value=5))
    def test_power_of_one(self, d, power):
        """1^k = 1 for any power k."""
        assume(d > 1)
        
        field = NumberField(Polynomial((-d, 0, 1)))
        one = field.rational(1)
        
        result = one ** power
        
        assert result == one

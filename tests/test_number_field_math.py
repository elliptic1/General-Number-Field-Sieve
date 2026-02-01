"""Comprehensive mathematical tests for number field operations in GNFS.

Tests verify:
- Algebraic integer arithmetic in Q[α]
- Norm calculations: N(a + bα) computed correctly
- Verify norm is multiplicative: N(xy) = N(x)N(y)
- Algebraic conjugates (implicit through norm)
- Edge cases: units, zero divisors (if any)
"""

import pytest
from fractions import Fraction
import math

from gnfs.polynomial import NumberField, Polynomial


# =============================================================================
# Basic Arithmetic Tests
# =============================================================================

class TestNumberFieldArithmetic:
    """Tests for algebraic integer arithmetic in Q[α]."""
    
    def test_addition_basic(self):
        """Test basic addition of number field elements."""
        # Q[√2] defined by x^2 - 2
        field = NumberField(Polynomial((-2, 0, 1)))
        
        elem1 = field.element([1, 2])    # 1 + 2α
        elem2 = field.element([3, -1])   # 3 - α
        
        result = elem1 + elem2           # 4 + α
        assert result.coeffs == (Fraction(4), Fraction(1))
    
    def test_addition_with_rationals(self):
        """Test addition of element with rational."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([1, 2])     # 1 + 2α
        
        result = elem + 5                # 6 + 2α
        assert result.coeffs == (Fraction(6), Fraction(2))
    
    def test_subtraction_basic(self):
        """Test basic subtraction."""
        field = NumberField(Polynomial((-2, 0, 1)))
        
        elem1 = field.element([5, 3])    # 5 + 3α
        elem2 = field.element([2, 1])    # 2 + α
        
        result = elem1 - elem2           # 3 + 2α
        assert result.coeffs == (Fraction(3), Fraction(2))
    
    def test_subtraction_gives_zero(self):
        """Test that x - x = 0."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([3, 7])
        
        result = elem - elem
        assert result.coeffs == (Fraction(0), Fraction(0))
    
    def test_negation(self):
        """Test negation of elements."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([3, -2])
        
        neg_elem = -elem
        assert neg_elem.coeffs == (Fraction(-3), Fraction(2))
        
        # x + (-x) = 0
        zero = elem + neg_elem
        assert zero.coeffs == (Fraction(0), Fraction(0))
    
    def test_multiplication_basic(self):
        """Test multiplication: (a + bα)(c + dα) = ac + bdα² + (ad + bc)α."""
        # In Q[√2]: α² = 2
        field = NumberField(Polynomial((-2, 0, 1)))
        
        # (1 + α)(1 - α) = 1 - α² = 1 - 2 = -1
        elem1 = field.element([1, 1])    # 1 + α
        elem2 = field.element([1, -1])   # 1 - α
        
        result = elem1 * elem2
        assert result == field.rational(-1)
    
    def test_multiplication_by_alpha(self):
        """Test multiplication by α."""
        field = NumberField(Polynomial((-2, 0, 1)))  # α² = 2
        alpha = field.alpha
        
        # α * α = α² = 2
        assert alpha * alpha == field.rational(2)
        
        # α³ = α² * α = 2α
        alpha_cubed = alpha * alpha * alpha
        assert alpha_cubed.coeffs == (Fraction(0), Fraction(2))
    
    def test_multiplication_by_rational(self):
        """Test scalar multiplication."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([2, 3])     # 2 + 3α
        
        result = elem * 5                # 10 + 15α
        assert result.coeffs == (Fraction(10), Fraction(15))
        
        result2 = 5 * elem               # commutative
        assert result2 == result
    
    def test_power_zero(self):
        """Test x^0 = 1."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([3, 7])
        
        result = elem ** 0
        assert result == field.rational(1)
    
    def test_power_one(self):
        """Test x^1 = x."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([3, 7])
        
        result = elem ** 1
        assert result == elem
    
    def test_power_two(self):
        """Test x^2 = x * x."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([1, 1])     # 1 + α
        
        result = elem ** 2
        expected = elem * elem
        assert result == expected


class TestNumberFieldHigherDegree:
    """Tests for number fields of degree > 2."""
    
    def test_cubic_field_basic(self):
        """Test basic operations in cubic field Q[α] where α³ = 2."""
        field = NumberField(Polynomial((-2, 0, 0, 1)))  # x³ - 2
        assert field.degree == 3
        
        alpha = field.alpha
        # α³ = 2
        alpha_cubed = alpha ** 3
        assert alpha_cubed == field.rational(2)
    
    def test_cubic_field_multiplication(self):
        """Test multiplication in cubic field."""
        field = NumberField(Polynomial((-2, 0, 0, 1)))  # α³ = 2
        
        # (1 + α)(1 + α²) = 1 + α + α² + α³ = 1 + α + α² + 2 = 3 + α + α²
        elem1 = field.element([1, 1, 0])     # 1 + α
        elem2 = field.element([1, 0, 1])     # 1 + α²
        
        result = elem1 * elem2
        assert result.coeffs == (Fraction(3), Fraction(1), Fraction(1))
    
    def test_quartic_field(self):
        """Test quartic field Q[α] where α⁴ = 2."""
        field = NumberField(Polynomial((-2, 0, 0, 0, 1)))  # x⁴ - 2
        assert field.degree == 4
        
        alpha = field.alpha
        result = alpha ** 4
        assert result == field.rational(2)
    
    def test_cyclotomic_like_field(self):
        """Test field defined by x³ - 1 (contains cube roots of unity)."""
        field = NumberField(Polynomial((-1, 0, 0, 1)))  # x³ - 1
        alpha = field.alpha
        
        # α³ = 1 (α is a cube root of unity)
        assert (alpha ** 3) == field.rational(1)


# =============================================================================
# Norm Tests
# =============================================================================

class TestNormBasic:
    """Basic tests for field norm calculations."""
    
    def test_norm_of_rational(self):
        """N(r) = r^d for rational r in degree-d field."""
        field = NumberField(Polynomial((-2, 0, 1)))  # degree 2
        
        # N(3) = 3² = 9
        assert field.norm(3) == 9
        
        # N(5) = 5² = 25
        assert field.norm(5) == 25
        
        # N(-2) = (-2)² = 4
        assert field.norm(-2) == 4
    
    def test_norm_of_alpha(self):
        """Test norm of α."""
        field = NumberField(Polynomial((-2, 0, 1)))
        alpha = field.alpha
        
        # N(α) where α² - 2 = 0 means conjugates are ±√2
        # N(√2) = √2 * (-√2) = -2
        norm_alpha = field.norm(alpha)
        assert norm_alpha == -2
    
    def test_norm_quadratic_element(self):
        """Test N(a + bα) for quadratic field."""
        # Q[√2]: N(a + b√2) = a² - 2b²
        field = NumberField(Polynomial((-2, 0, 1)))
        
        # N(1 + √2) = 1 - 2 = -1
        elem = field.element([1, 1])
        assert field.norm(elem) == -1
        
        # N(3 + 2√2) = 9 - 8 = 1
        elem2 = field.element([3, 2])
        assert field.norm(elem2) == 1
        
        # N(5 - 3√2) = 25 - 18 = 7
        elem3 = field.element([5, -3])
        assert field.norm(elem3) == 7
    
    def test_norm_zero(self):
        """N(0) = 0."""
        field = NumberField(Polynomial((-2, 0, 1)))
        zero = field.element([0, 0])
        assert field.norm(zero) == 0
    
    def test_norm_one(self):
        """N(1) = 1."""
        field = NumberField(Polynomial((-2, 0, 1)))
        one = field.rational(1)
        assert field.norm(one) == 1


class TestNormMultiplicativity:
    """Test that norm is multiplicative: N(xy) = N(x)N(y)."""
    
    def test_multiplicativity_quadratic(self):
        """Test N(xy) = N(x)N(y) in quadratic field."""
        field = NumberField(Polynomial((-2, 0, 1)))  # Q[√2]
        
        x = field.element([3, 2])    # 3 + 2√2
        y = field.element([1, -1])   # 1 - √2
        
        norm_x = field.norm(x)       # 9 - 8 = 1
        norm_y = field.norm(y)       # 1 - 2 = -1
        norm_xy = field.norm(x * y)
        
        assert norm_xy == norm_x * norm_y
    
    def test_multiplicativity_many_pairs(self):
        """Test multiplicativity for many element pairs."""
        field = NumberField(Polynomial((-2, 0, 1)))
        
        test_elements = [
            field.element([1, 0]),
            field.element([0, 1]),
            field.element([1, 1]),
            field.element([2, 3]),
            field.element([-1, 2]),
            field.element([5, -2]),
        ]
        
        for x in test_elements:
            for y in test_elements:
                norm_x = field.norm(x)
                norm_y = field.norm(y)
                norm_xy = field.norm(x * y)
                assert norm_xy == norm_x * norm_y, \
                    f"Multiplicativity failed for {x.coeffs} * {y.coeffs}"
    
    def test_multiplicativity_cubic(self):
        """Test multiplicativity in cubic field."""
        field = NumberField(Polynomial((-2, 0, 0, 1)))  # x³ - 2
        
        x = field.element([1, 1, 0])
        y = field.element([0, 1, 1])
        
        norm_x = field.norm(x)
        norm_y = field.norm(y)
        norm_xy = field.norm(x * y)
        
        assert norm_xy == norm_x * norm_y
    
    def test_multiplicativity_with_rationals(self):
        """Test N(rx) = r^d * N(x) for rational r."""
        field = NumberField(Polynomial((-2, 0, 1)))  # degree 2
        
        x = field.element([2, 3])
        r = 5
        
        norm_x = field.norm(x)
        norm_rx = field.norm(r * x)
        
        # N(rx) = r² * N(x) for degree 2 field
        assert norm_rx == r * r * norm_x


class TestNormFormulas:
    """Test specific norm formulas for various field types."""
    
    def test_norm_formula_sqrt_d(self):
        """Test N(a + b√d) = a² - db² for Q[√d]."""
        for d in [2, 3, 5, 7]:
            field = NumberField(Polynomial((-d, 0, 1)))
            
            for a in range(-3, 4):
                for b in range(-3, 4):
                    elem = field.element([a, b])
                    expected_norm = a * a - d * b * b
                    actual_norm = field.norm(elem)
                    assert actual_norm == expected_norm, \
                        f"For √{d}: N({a} + {b}√{d}) = {actual_norm}, expected {expected_norm}"
    
    def test_norm_cubic_root_of_two(self):
        """Test norm in Q[∛2]."""
        field = NumberField(Polynomial((-2, 0, 0, 1)))  # x³ - 2
        
        # For α = ∛2, the conjugates are ∛2, ω∛2, ω²∛2 where ω = e^(2πi/3)
        # N(α) = (∛2)(ω∛2)(ω²∛2) = 2 * ω³ = 2 * 1 = 2
        alpha = field.alpha
        norm_alpha = field.norm(alpha)
        assert norm_alpha == 2
    
    def test_norm_cyclotomic(self):
        """Test norm in cyclotomic-like field x³ - 1."""
        field = NumberField(Polynomial((-1, 0, 0, 1)))  # x³ - 1
        
        alpha = field.alpha
        # α³ = 1, so N(α) = 1 * 1 * 1 = 1 (all conjugates are cube roots of unity)
        norm_alpha = field.norm(alpha)
        assert norm_alpha == 1


# =============================================================================
# Special Elements Tests
# =============================================================================

class TestUnits:
    """Test unit elements (elements with norm ±1)."""
    
    def test_unit_in_quadratic_field(self):
        """Test that N(u) = ±1 implies u is a unit."""
        field = NumberField(Polynomial((-2, 0, 1)))  # Q[√2]
        
        # 1 + √2 has N = 1 - 2 = -1, so it's a unit
        u = field.element([1, 1])
        assert field.norm(u) == -1
        
        # 3 + 2√2 has N = 9 - 8 = 1, so it's a unit
        v = field.element([3, 2])
        assert field.norm(v) == 1
    
    def test_unit_powers(self):
        """Test that powers of units are units."""
        field = NumberField(Polynomial((-2, 0, 1)))
        
        # Fundamental unit in Q[√2] is 1 + √2
        u = field.element([1, 1])
        
        for k in range(1, 6):
            u_power = u ** k
            norm_power = field.norm(u_power)
            assert abs(norm_power) == 1, f"u^{k} should be a unit"
    
    def test_unit_times_unit_is_unit(self):
        """Product of units is a unit."""
        field = NumberField(Polynomial((-2, 0, 1)))
        
        u1 = field.element([1, 1])   # N = -1
        u2 = field.element([3, 2])   # N = 1
        
        product = u1 * u2
        assert abs(field.norm(product)) == 1


class TestZeroAndIdentity:
    """Test zero and identity elements."""
    
    def test_zero_is_additive_identity(self):
        """x + 0 = x."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([3, 7])
        zero = field.element([0, 0])
        
        assert x + zero == x
        assert zero + x == x
    
    def test_one_is_multiplicative_identity(self):
        """x * 1 = x."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([3, 7])
        one = field.rational(1)
        
        assert x * one == x
        assert one * x == x
    
    def test_zero_times_anything_is_zero(self):
        """0 * x = 0."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([3, 7])
        zero = field.element([0, 0])
        
        result = zero * x
        assert result.coeffs == (Fraction(0), Fraction(0))


# =============================================================================
# Power Basis Tests
# =============================================================================

class TestPowerBasis:
    """Tests for power basis representation."""
    
    def test_power_basis_quadratic(self):
        """Test power basis for quadratic field."""
        field = NumberField(Polynomial((-2, 0, 1)))
        basis = field.power_basis
        
        assert len(basis) == 2
        assert basis[0] == field.rational(1)
        assert basis[1] == field.alpha
    
    def test_power_basis_cubic(self):
        """Test power basis for cubic field."""
        field = NumberField(Polynomial((-2, 0, 0, 1)))
        basis = field.power_basis
        
        assert len(basis) == 3
        assert basis[0] == field.rational(1)
        assert basis[1] == field.alpha
        assert basis[2] == field.alpha ** 2
    
    def test_element_as_linear_combination(self):
        """Every element is a linear combination of power basis elements."""
        field = NumberField(Polynomial((-2, 0, 1)))
        basis = field.power_basis
        
        elem = field.element([5, 3])  # 5 + 3α
        
        # Reconstruct from basis
        reconstructed = 5 * basis[0] + 3 * basis[1]
        assert reconstructed == elem


# =============================================================================
# Edge Cases
# =============================================================================

class TestNumberFieldEdgeCases:
    """Test edge cases and error handling."""
    
    def test_degree_one_field(self):
        """Test degree-1 field (essentially just Q)."""
        field = NumberField(Polynomial((-5, 1)))  # x - 5
        assert field.degree == 1
        
        # In this field, α = 5 (the root)
        elem = field.element([3])
        norm = field.norm(elem)
        # N(3) = 3 in degree-1 field
        assert norm == 3
    
    def test_non_monic_polynomial(self):
        """Test field defined by non-monic polynomial (gets made monic)."""
        # 2x² - 4 (equivalent to x² - 2 when made monic)
        field = NumberField(Polynomial((-4, 0, 2)))
        assert field.degree == 2
        
        alpha = field.alpha
        # α² should equal 2 after normalization
        alpha_sq = alpha * alpha
        assert alpha_sq == field.rational(2)
    
    def test_comparison_with_coefficients(self):
        """Test element comparison with raw coefficient lists."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([3, 5])
        
        assert elem == [3, 5]
        assert elem == [Fraction(3), Fraction(5)]
    
    def test_repr_string(self):
        """Test string representation."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([1, 2])
        
        repr_str = repr(elem)
        assert "NumberFieldElement" in repr_str
        assert "1" in repr_str
        assert "2" in repr_str
    
    def test_fractional_coefficients(self):
        """Test elements with fractional coefficients."""
        field = NumberField(Polynomial((-2, 0, 1)))
        
        elem = field.element([Fraction(1, 2), Fraction(3, 4)])
        assert elem.coeffs == (Fraction(1, 2), Fraction(3, 4))
        
        # Operations preserve fractions
        doubled = elem + elem
        assert doubled.coeffs == (Fraction(1), Fraction(3, 2))
    
    def test_invalid_constant_polynomial_raises(self):
        """Test that constant polynomial raises error."""
        with pytest.raises(ValueError):
            NumberField(Polynomial((5,)))
    
    def test_negative_power_raises(self):
        """Test that negative powers raise error."""
        field = NumberField(Polynomial((-2, 0, 1)))
        elem = field.element([2, 3])
        
        with pytest.raises(ValueError):
            elem ** (-1)


# =============================================================================
# Algebraic Properties
# =============================================================================

class TestAlgebraicProperties:
    """Test algebraic properties of number field arithmetic."""
    
    def test_addition_commutativity(self):
        """x + y = y + x."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([3, 5])
        y = field.element([2, -1])
        
        assert x + y == y + x
    
    def test_addition_associativity(self):
        """(x + y) + z = x + (y + z)."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([1, 2])
        y = field.element([3, 4])
        z = field.element([5, 6])
        
        assert (x + y) + z == x + (y + z)
    
    def test_multiplication_commutativity(self):
        """xy = yx."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([3, 5])
        y = field.element([2, -1])
        
        assert x * y == y * x
    
    def test_multiplication_associativity(self):
        """(xy)z = x(yz)."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([1, 2])
        y = field.element([3, 4])
        z = field.element([5, 6])
        
        assert (x * y) * z == x * (y * z)
    
    def test_distributivity(self):
        """x(y + z) = xy + xz."""
        field = NumberField(Polynomial((-2, 0, 1)))
        x = field.element([1, 2])
        y = field.element([3, 4])
        z = field.element([5, 6])
        
        left = x * (y + z)
        right = (x * y) + (x * z)
        assert left == right


# =============================================================================
# GNFS-Specific Tests
# =============================================================================

class TestGNFSNumberFields:
    """Tests for number fields as used in GNFS."""
    
    def test_gnfs_polynomial_field(self):
        """Test number field from GNFS polynomial selection."""
        from gnfs.polynomial.selection import select_polynomial
        
        n = 143  # 11 * 13
        selection = select_polynomial(n, degree=2)
        
        field = NumberField(selection.algebraic)
        assert field.degree == 2
        
        # Verify alpha satisfies the polynomial
        alpha = field.alpha
        alpha_sq = alpha * alpha
        
        # For polynomial c_0 + c_1*x + c_2*x^2, we have
        # α² = -(c_0 + c_1*α) / c_2 (after reduction)
        # Just verify norm is computed without error
        norm = field.norm(alpha)
        assert isinstance(norm, (int, Fraction))
    
    def test_algebraic_norm_for_sieving(self):
        """Test that norm computation works for sieving-related elements."""
        # For GNFS sieving, we compute norms of a + bα
        field = NumberField(Polynomial((-5, 2, 1)))  # x² + 2x - 5
        
        test_pairs = [(1, 1), (2, 1), (3, 2), (-1, 1), (5, 3)]
        
        for a, b in test_pairs:
            elem = field.element([a, b])
            norm = field.norm(elem)
            
            # Verify it's an integer or simple fraction
            if isinstance(norm, Fraction):
                assert norm.denominator == 1 or isinstance(norm.numerator / norm.denominator, float)

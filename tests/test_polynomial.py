"""Tests for polynomial utilities in GNFS."""

import pytest

from gnfs.polynomial.polynomial import Polynomial
from gnfs.polynomial.selection import (
    PolynomialSelection, 
    select_polynomial,
    select_polynomial_classic,
)


def test_polynomial_evaluate_and_degree():
    poly = Polynomial((1, 2, 3))
    assert poly.degree() == 2
    assert poly.evaluate(0) == 1
    assert poly.evaluate(1) == 6
    assert poly.evaluate(2) == 17


def test_select_polynomial_degree_one():
    selection = select_polynomial(10, degree=1)
    assert isinstance(selection, PolynomialSelection)
    assert selection.algebraic.coeffs == (-int(10 ** 0.5), 1)
    assert selection.rational.coeffs == (-int(10 ** 0.5), 1)


def test_select_polynomial_higher_degree():
    """Test that higher degree polynomials satisfy f(m) = n.
    
    The improved selection uses base-m expansion which produces
    polynomial f where f(m) = n exactly, ensuring the correct
    GNFS root property: f(m) ≡ 0 (mod n).
    """
    selection = select_polynomial(10, degree=3)
    assert isinstance(selection, PolynomialSelection)
    poly = selection.algebraic
    assert poly.degree() == 3
    
    # The key property: f(m) = n (so f(m) ≡ 0 mod n)
    assert poly.evaluate(selection.m) == 10


def test_select_polynomial_classic_higher_degree():
    """Test the classic (x + m)^d - n construction for comparison.
    
    The classic method produces f(-m) ≡ 0 (mod n), which is a different
    root convention. This is kept for backward compatibility.
    """
    selection = select_polynomial_classic(10, degree=3)
    assert isinstance(selection, PolynomialSelection)
    poly = selection.algebraic
    assert poly.degree() == 3
    
    # For n=10 and degree=3, m = round(10^(1/3)) = 2
    # (x + 2)^3 - 10 = x^3 + 6x^2 + 12x + 8 - 10 = x^3 + 6x^2 + 12x - 2
    # Coefficients stored lowest to highest: (-2, 12, 6, 1)
    assert poly.coeffs == (-2, 12, 6, 1)


def test_select_polynomial_invalid_degree():
    with pytest.raises(ValueError):
        select_polynomial(10, degree=0)


def test_polynomial_constant_degree_zero():
    poly = Polynomial((3,))
    assert poly.degree() == 0
    assert poly.evaluate(7) == 3


def test_select_polynomial_root_property():
    """Test the correct GNFS root property.
    
    For GNFS, both polynomials must share a common root m modulo n:
    - f(m) = n (so f(m) ≡ 0 mod n)
    - g(m) = 0 (rational polynomial)
    
    This ensures the homomorphism φ: Z[x]/(f(x)) → Z[x]/(g(x)) 
    can be used to find relations.
    """
    n = 10
    selection = select_polynomial(n, degree=2)
    m = selection.m
    
    # f(m) = n (correct GNFS property)
    assert selection.algebraic.evaluate(m) == n
    
    # g(m) = 0
    assert selection.rational.evaluate(m) == 0


def test_select_polynomial_classic_root_property():
    """Test root property for classic selection.
    
    The classic (x + m)^d - n construction has root at -m:
    f(-m) = 0 - n = -n ≡ 0 (mod n)
    """
    n = 10
    selection = select_polynomial_classic(n, degree=2)
    m = selection.m
    
    # Classic method: f(-m) ≡ 0 (mod n)
    assert selection.algebraic.evaluate(-m) % n == 0
    
    # g(m) = 0
    assert selection.rational.evaluate(m) == 0

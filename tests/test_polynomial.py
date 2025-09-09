import pytest
from gnfs.polynomial.polynomial import Polynomial
from gnfs.polynomial.selection import select_polynomial


def test_polynomial_evaluate_and_degree():
    poly = Polynomial((1, 2, 3))
    assert poly.degree() == 2
    assert poly.evaluate(0) == 1
    assert poly.evaluate(1) == 6
    assert poly.evaluate(2) == 17


def test_select_polynomial_degree_one():
    poly = select_polynomial(10, degree=1)
    assert isinstance(poly, Polynomial)
    assert poly.coeffs == (-int(10 ** 0.5), 1)


def test_select_polynomial_higher_degree():
    poly = select_polynomial(10, degree=3)
    assert isinstance(poly, Polynomial)
    assert poly.degree() == 3
    # For n=10 and degree=3 we expect the (x + m)^3 - n construction
    # where m = round(n ** (1/3)) = 2, leading to coefficients
    # (-2, 12, 6, 1)
    assert poly.coeffs == (-2, 12, 6, 1)


def test_select_polynomial_invalid_degree():
    with pytest.raises(ValueError):
        select_polynomial(10, degree=0)


def test_polynomial_constant_degree_zero():
    poly = Polynomial((3,))
    assert poly.degree() == 0
    assert poly.evaluate(7) == 3


def test_select_polynomial_root_property():
    n = 10
    poly = select_polynomial(n, degree=2)
    m = round(n ** 0.5)
    assert poly.evaluate(-m) % n == 0

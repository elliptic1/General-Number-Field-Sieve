import pytest
from gnfs.polynomial import Polynomial, select_polynomial


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
    assert poly.coeffs == (-10, 0, 0, 1)

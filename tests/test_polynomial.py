import pytest
from gnfs.polynomial import Polynomial, select_polynomial


def test_polynomial_evaluate_and_degree():
    poly = Polynomial((1, 2, 3))
    assert poly.degree() == 2
    assert poly.evaluate(0) == 1
    assert poly.evaluate(1) == 6
    assert poly.evaluate(2) == 17


def test_select_polynomial():
    poly = select_polynomial(10)
    assert isinstance(poly, Polynomial)
    assert poly.coeffs == (-int(10 ** 0.5), 1)

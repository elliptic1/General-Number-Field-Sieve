from fractions import Fraction

import pytest

from gnfs.polynomial import NumberField, Polynomial


def test_number_field_basic_arithmetic():
    field = NumberField(Polynomial((-2, 0, 1)))  # x^2 - 2
    alpha = field.alpha
    assert field.degree == 2
    assert alpha * alpha == field.rational(2)
    assert (alpha + 1) * (alpha - 1) == field.rational(1)


def test_number_field_norm():
    field = NumberField(Polynomial((-2, 0, 1)))
    alpha = field.alpha
    assert field.norm(alpha + 1) == -1
    assert field.norm(3) == 9


def test_number_field_coefficients_and_comparison():
    field = NumberField(Polynomial((-1, 0, 0, 1)))  # x^3 - 1
    alpha = field.alpha
    element = field.element([Fraction(1, 2), 0, Fraction(1, 2)])
    # Ensure equality works with raw coefficient iterables and rationals
    assert element == [Fraction(1, 2), 0, Fraction(1, 2)]
    assert alpha == [0, 1, 0]
    assert field.norm(alpha) == 1


def test_number_field_requires_non_constant_polynomial():
    with pytest.raises(ValueError):
        NumberField(Polynomial((5,)))

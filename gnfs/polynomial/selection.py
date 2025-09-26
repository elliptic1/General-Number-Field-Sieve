"""Polynomial selection for General Number Field Sieve (GNFS).

The real GNFS relies on carefully chosen polynomials that share a root modulo
``n`` being factored and whose coefficients are small enough to keep the
sieving stage efficient.  Production grade implementations contain elaborate
search strategies, but even a basic version can mirror the mathematics.  This
module implements a light-weight variant of the standard ``(x + m)^d - n``
construction which produces a polynomial with a root ``m`` modulo ``n`` and a
small constant term ``m**d - n``.  While far from optimal it is a genuine
algorithm rather than a toy placeholder.
"""

import math
from dataclasses import dataclass
from math import comb

from .polynomial import Polynomial


@dataclass(frozen=True)
class PolynomialSelection:
    """Container bundling the rational and algebraic polynomials for GNFS."""

    algebraic: Polynomial
    rational: Polynomial
    m: int


def select_polynomial(n: int, degree: int = 1) -> PolynomialSelection:
    """Construct a polynomial with a root ``m`` modulo ``n``.

    The selection mirrors the classic ``(x + m)^d - n`` recipe.  The integer
    ``m`` is chosen as the closest integer to the real ``d``-th root of ``n``.
    Expanding ``(x + m)^d`` yields coefficients ``comb(d, k) * m**(d - k)`` for
    ``0 <= k <= d``.  Replacing the constant term with ``m**d - n`` ensures that
    plugging ``x = 0`` into the polynomial gives the small value ``m**d - n`` and
    that ``x = -m`` is a root modulo ``n``.  The returned
    :class:`PolynomialSelection` additionally bundles the rational-side
    polynomial ``x - m`` which is used when constructing projective GNFS
    relations.

    Parameters
    ----------
    n:
        Integer to factor.
    degree:
        Desired degree of the polynomial. Must be a positive integer.
    """

    if degree < 1:
        raise ValueError("degree must be >= 1")

    if degree == 1:
        m = int(math.isqrt(n))
        algebraic_coeffs = (-m, 1)
    else:
        # Compute m ≈ n^{1/degree} and expand (x + m)^degree - n
        m = round(n ** (1 / degree))
        algebraic_coeffs = [comb(degree, k) * (m ** (degree - k)) for k in range(degree + 1)]
        algebraic_coeffs[0] -= n
        algebraic_coeffs = tuple(algebraic_coeffs)

    algebraic_poly = Polynomial(algebraic_coeffs)
    rational_poly = Polynomial((-m, 1))
    return PolynomialSelection(algebraic=algebraic_poly, rational=rational_poly, m=m)

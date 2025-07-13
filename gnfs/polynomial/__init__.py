"""Polynomial selection for General Number Field Sieve (GNFS).

This module defines utilities for generating polynomials used in the GNFS
algorithm. The implementation provided here is intentionally simplistic and
serves primarily as a placeholder for a more sophisticated approach.
"""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Polynomial:
    """Represents a polynomial used in GNFS."""

    coeffs: Tuple[int, ...]

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def evaluate(self, x: int) -> int:
        result = 0
        for power, coeff in enumerate(self.coeffs):
            result += coeff * (x ** power)
        return result


def select_polynomial(n: int, degree: int = 1) -> Polynomial:
    """Return a basic polynomial for the given ``n`` and ``degree``.

    This routine provides a minimal placeholder for GNFS polynomial
    selection.  For degree one it chooses ``x - floor(sqrt(n))``.  For
    higher degrees it returns ``x**degree - n`` which at least matches
    the required degree even though it lacks the sophisticated search
    normally used in production implementations.

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
        coeffs = (-int(math.isqrt(n)), 1)
    else:
        coeffs = [-n] + [0] * (degree - 1) + [1]
        coeffs = tuple(coeffs)

    return Polynomial(coeffs)

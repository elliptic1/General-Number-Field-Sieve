"""Polynomial selection for General Number Field Sieve (GNFS).

This module defines utilities for generating polynomials used in the GNFS
algorithm. The implementation provided here is intentionally simplistic and
serves primarily as a placeholder for a more sophisticated approach.
"""

from dataclasses import dataclass
from typing import Tuple


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
    """Return a simple polynomial for demonstration purposes.

    Parameters
    ----------
    n:
        Integer to factor.
    degree:
        Desired degree of the polynomial.  The current toy implementation only
        supports degree one and ignores this parameter.
    """
    # In a full GNFS implementation this step is highly complex. Here we return
    # x - sqrt(n) as a trivial placeholder regardless of ``degree``.
    coeffs = (-int(n ** 0.5), 1)
    return Polynomial(coeffs)

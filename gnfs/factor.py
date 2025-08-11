"""Main GNFS factorisation interface.

This module glues together the individual stages implemented throughout the
package: polynomial selection, sieving, linear algebra and the square root
step.  Although the surrounding modules are purposely lightweight, they now
reflect the genuine algorithms behind GNFS and this function therefore runs a
true albeit small‑scale instance of the sieve."""

from typing import Iterable, List

import sympy as sp

from .polynomial import select_polynomial
from .sieve import find_relations
from .sqrt import find_factors


def gnfs_factor(
    n: int, bound: int = 30, interval: int = 50, degree: int = 1
) -> Iterable[int]:
    """Attempt to factor ``n`` using the GNFS pipeline.

    Parameters
    ----------
    n:
        Integer to factor.
    bound:
        Smoothness bound for the factor base.
    interval:
        Sieving interval radius.
    degree:
        Degree of the algebraic polynomial used in the sieve.
    """

    poly = select_polynomial(n, degree=degree)
    primes: List[int] = list(sp.primerange(2, bound + 1))
    relations = list(find_relations(poly, primes=primes, interval=interval))
    return list(find_factors(n, relations, primes))

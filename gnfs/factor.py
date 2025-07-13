"""Main GNFS factorization interface."""

from typing import Iterable, List

import sympy as sp

from .polynomial import select_polynomial
from .sieve import find_relations
from .sqrt import find_factors


def gnfs_factor(
    n: int, bound: int = 30, interval: int = 50, degree: int = 1
) -> Iterable[int]:
    """Attempt to factor ``n`` using a very small GNFS pipeline."""

    poly = select_polynomial(n, degree=degree)
    primes: List[int] = list(sp.primerange(2, bound + 1))
    relations = list(find_relations(poly, primes=primes, interval=interval))
    return list(find_factors(n, relations, primes))

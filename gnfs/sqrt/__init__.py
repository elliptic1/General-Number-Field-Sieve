"""Square root step for the GNFS.

Given a set of relations and their prime factorizations, this step
combines them according to dependencies found by the linear algebra
phase.  From the resulting congruence of squares a non-trivial factor
of ``n`` is extracted using ``gcd``.
"""

from math import isqrt
from typing import Iterable, List

import sympy as sp

from ..linalg import solve_matrix
from ..sieve import Relation


def find_factors(n: int, relations: Iterable[Relation], primes: List[int]) -> Iterable[int]:
    """Attempt to recover a factor of ``n`` from ``relations``."""
    rel_list = list(relations)
    for dep in solve_matrix(rel_list, primes):
        if not dep:
            continue
        x = 1
        prod = 1
        for idx in dep:
            rel = rel_list[idx]
            x = (x * rel.a) % n
            prod *= abs(rel.value)
        y = isqrt(prod)
        if y * y != prod:
            # Should not happen if exponents are even, but be safe
            continue
        g = sp.gcd(x - y, n)
        if 1 < g < n:
            yield g
            yield n // g
            return

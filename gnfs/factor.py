"""Main GNFS factorisation interface.

This module glues together the individual stages implemented throughout the
package: polynomial selection, sieving, linear algebra and the square root
step.  Although the surrounding modules are purposely lightweight, they now
reflect the genuine algorithms behind GNFS and this function therefore runs a
true albeit small‑scale instance of the sieve."""

from typing import Iterable, List, Set, Tuple

import sympy as sp

from .polynomial import select_polynomial
from .sieve import Relation, find_relations
from .sqrt import find_factors


def gnfs_factor(
    n: int, bound: int = 30, interval: int = 50, degree: int = 1, max_rounds: int = 5
) -> List[int]:
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
    max_rounds:
        Maximum number of sieving expansions to attempt when gathering relations.
    """

    selection = select_polynomial(n, degree=degree)
    primes: List[int] = list(sp.primerange(2, bound + 1))

    required_relations = len(primes) + 1
    relations: List[Relation] = []
    seen: Set[Tuple[int, int]] = set()
    current_interval = interval

    for _ in range(max_rounds):
        for rel in find_relations(selection, primes=primes, interval=current_interval):
            key = (rel.a, rel.b)
            if key in seen:
                continue
            seen.add(key)
            relations.append(rel)
            if len(relations) >= required_relations:
                break
        if len(relations) >= required_relations:
            break
        current_interval += interval

    if len(relations) < required_relations:
        return []

    return list(find_factors(n, relations, primes))

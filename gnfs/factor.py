"""Main GNFS factorization interface."""

from typing import Iterable

from .polynomial import select_polynomial
from .sieve import find_relations
from .square_root import find_factors


def gnfs_factor(n: int) -> Iterable[int]:
    """Attempt to factor ``n`` using a toy GNFS implementation."""
    poly = select_polynomial(n)
    relations = list(find_relations(poly))
    return list(find_factors(n, relations))

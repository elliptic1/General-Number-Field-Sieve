"""Square root step for the GNFS."""

from typing import Iterable
from .linear_algebra import solve_matrix
from .sieve import Relation


def find_factors(n: int, relations: Iterable[Relation]) -> Iterable[int]:
    """Return dummy factors of n."""
    # Real GNFS would combine relations to produce squares and compute the
    # gcd. Here we return trivial factors for demonstration.
    if n % 2 == 0:
        yield 2
        yield n // 2

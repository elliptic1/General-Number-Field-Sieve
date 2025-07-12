"""Square root step for the GNFS."""

"""Square root step for the GNFS.

This file contains a drastically simplified stand in for the square root phase
of the General Number Field Sieve.  In a full implementation this step would
combine the collected relations, solve a sparse matrix and finally compute the
gcd to obtain non–trivial factors.  For the purposes of this toy project we
instead perform a naive trial division.  This keeps the code extremely short
while still providing behaviour that is a bit more realistic for the command
line demo and the unit tests.
"""

from typing import Iterable

from .linear_algebra import solve_matrix  # noqa: F401  imported for parity with real GNFS
from .sieve import Relation


def find_factors(n: int, relations: Iterable[Relation]) -> Iterable[int]:
    """Return small non–trivial factors of ``n`` using trial division.

    Parameters
    ----------
    n:
        Integer to factor.
    relations:
        Collected relations from the sieving step (ignored here).

    Yields
    ------
    int
        A non–trivial factor of ``n`` and its complementary cofactor.
    """

    # Attempt to find a factor by simple trial division.  This keeps the code
    # readable and is sufficient for very small inputs which is all this
    # demonstration aims to handle.
    if n < 2:
        return
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            yield p
            yield n // p
            return

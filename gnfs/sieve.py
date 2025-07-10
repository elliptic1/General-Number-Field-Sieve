"""Sieving step for the General Number Field Sieve (GNFS).

The real GNFS algorithm uses lattice sieving or line sieving to find smooth
relations. This simplified module only simulates the process by generating
pseudo-relations for a demonstration of code structure.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

from .polynomial import Polynomial


@dataclass
class Relation:
    """Represents a smooth relation."""

    a: int
    b: int
    value: int


def find_relations(poly: Polynomial, bound: int = 100) -> Iterable[Relation]:
    """Yield dummy relations for demonstration purposes."""
    for a in range(1, 5):
        b = poly.evaluate(a)
        yield Relation(a=a, b=b, value=a * b)

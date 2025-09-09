"""Polynomial representation for GNFS."""

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

"""Relation representation for the GNFS sieving step."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Relation:
    """Represents a smooth relation discovered during sieving."""

    a: int
    b: int
    algebraic_value: int
    rational_value: int
    algebraic_factors: Dict[int, int]
    rational_factors: Dict[int, int]

    def combined_factors(self) -> Dict[int, int]:
        """Return exponent counts from both algebraic and rational sides."""

        combined: Dict[int, int] = dict(self.algebraic_factors)
        for prime, exponent in self.rational_factors.items():
            combined[prime] = combined.get(prime, 0) + exponent
        return combined

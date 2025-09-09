"""Relation representation for the GNFS sieving step."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Relation:
    """Represents a smooth relation discovered during sieving."""

    a: int
    b: int
    value: int
    factors: Dict[int, int]

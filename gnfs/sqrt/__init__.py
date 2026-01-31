"""Square root step for the GNFS.

This module provides square root computation for extracting factors.
The main entry point is find_factors(), which tries multiple methods:

1. Algebraic sqrt using number field arithmetic (when poly/m provided)
2. Simple integer sqrt of evaluated products (fallback)
"""

from .square_root import find_factors, find_factors_simple
from .algebraic_sqrt import (
    NumberFieldElement,
    AlgebraicSquareRoot,
    tonelli_shanks,
    sqrt_mod_prime_power,
)

__all__ = [
    "find_factors",
    "find_factors_simple",
    "NumberFieldElement",
    "AlgebraicSquareRoot",
    "tonelli_shanks",
    "sqrt_mod_prime_power",
]

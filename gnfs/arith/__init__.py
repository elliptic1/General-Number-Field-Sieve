"""Large number arithmetic with optional gmpy2 acceleration.

This module provides a unified interface for big integer operations,
using gmpy2 when available for 10-100x speedup, with pure Python fallback.

Usage:
    from gnfs.arith import mpz, isqrt, gcd, is_prime, next_prime, factor

The functions automatically use the fastest available implementation.
"""

from .bigint import (
    HAVE_GMPY2,
    mpz,
    isqrt,
    gcd,
    lcm,
    mod_inverse,
    jacobi,
    is_prime,
    next_prime,
    prev_prime,
    is_power,
    iroot,
    factor_trial,
    powmod,
)

__all__ = [
    "HAVE_GMPY2",
    "mpz",
    "isqrt",
    "gcd",
    "lcm",
    "mod_inverse",
    "jacobi",
    "is_prime",
    "next_prime",
    "prev_prime",
    "is_power",
    "iroot",
    "factor_trial",
    "powmod",
]

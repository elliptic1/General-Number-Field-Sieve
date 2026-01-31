"""Square root step for the GNFS.

Given a set of relations and their prime factorizations, this step
combines them according to dependencies found by the linear algebra
phase. From the resulting congruence of squares a non-trivial factor
of ``n`` is extracted using ``gcd``.

The key challenge is computing the algebraic square root: we have a
product β = ∏(a_i - b_i*α) that is a perfect square in Z[α], and we
need to find γ such that γ² = β, then evaluate γ(m) to get an integer.

This module provides:
1. Simple method using integer sqrt of evaluated product (original)
2. Algebraic method using number field arithmetic (improved)
"""

from math import gcd, isqrt
from typing import Iterable, List, Optional, Tuple

from ..linalg import solve_matrix
from ..sieve import Relation


def find_factors(
    n: int,
    relations: Iterable[Relation],
    primes: List[int],
    poly_coeffs: Optional[List[int]] = None,
    m: Optional[int] = None,
) -> Iterable[int]:
    """Attempt to recover a factor of ``n`` from ``relations``.
    
    Args:
        n: The number to factor
        relations: Smooth relations from sieving
        primes: Factor base primes
        poly_coeffs: Coefficients of algebraic polynomial (for improved sqrt)
        m: Integer root of polynomial mod n (for improved sqrt)
    
    Yields:
        Factors of n as they are found
    """
    rel_list = list(relations)
    
    for dep in solve_matrix(rel_list, primes):
        if not dep:
            continue
        
        # Try improved algebraic sqrt if parameters provided
        if poly_coeffs is not None and m is not None:
            factor = _try_algebraic_sqrt(n, rel_list, dep, poly_coeffs, m)
            if factor is not None:
                yield factor
                yield n // factor
                return
        
        # Fallback to simple method
        factor = _try_simple_sqrt(n, rel_list, dep)
        if factor is not None:
            yield factor
            yield n // factor
            return


def _try_simple_sqrt(
    n: int,
    relations: List[Relation],
    dependency: List[int],
) -> Optional[int]:
    """Original simple square root method.
    
    Computes x from rational side, y from integer sqrt of algebraic product.
    """
    x = 1
    prod = 1
    for idx in dependency:
        rel = relations[idx]
        x = (x * (rel.rational_value % n)) % n
        prod *= abs(rel.algebraic_value)
    
    y = isqrt(prod)
    if y * y != prod:
        # Not a perfect square - shouldn't happen with valid dependency
        return None
    
    y = y % n
    
    # Try both x - y and x + y
    for diff in [x - y, x + y]:
        g = gcd(diff % n, n)
        if 1 < g < n:
            return g
    
    return None


def _try_algebraic_sqrt(
    n: int,
    relations: List[Relation],
    dependency: List[int],
    poly_coeffs: List[int],
    m: int,
) -> Optional[int]:
    """Improved algebraic square root method.
    
    Uses number field arithmetic to properly compute the algebraic sqrt.
    """
    try:
        from .algebraic_sqrt import AlgebraicSquareRoot
        
        alg_sqrt = AlgebraicSquareRoot(poly_coeffs, m, n)
        result = alg_sqrt.extract_factor(relations, dependency)
        
        if result is not None:
            return result[0]
    except Exception:
        # Fall through to simple method on any error
        pass
    
    return None


def find_factors_simple(
    n: int,
    relations: Iterable[Relation],
    primes: List[int],
) -> Iterable[int]:
    """Simple factor finding using only integer arithmetic.
    
    This is the original method, kept for compatibility and as a fallback.
    """
    return find_factors(n, relations, primes, poly_coeffs=None, m=None)

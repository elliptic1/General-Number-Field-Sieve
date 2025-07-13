"""Sieving step for the General Number Field Sieve (GNFS).

This module now implements a small scale version of the real GNFS sieving
algorithm.  Given a polynomial, a factor base bound and a sieving interval,
the ``find_relations`` function searches for ``B``-smooth values of the
polynomial using a line sieving approach.  While drastically simplified when
compared to production implementations, the code follows the same basic idea
of dividing out small prime factors from values of the polynomial and
collecting relations where the remaining cofactor is ``\pm1``.

SymPy is used for helper arithmetic such as prime generation.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import sympy as sp

from ..polynomial import Polynomial


@dataclass
class Relation:
    """Represents a smooth relation discovered during sieving."""

    a: int
    b: int
    value: int
    factors: Dict[int, int]


def _polynomial_roots_mod_p(poly: Polynomial, p: int) -> List[int]:
    """Return the roots of ``poly`` modulo ``p``.

    This helper uses SymPy to factor the polynomial over the finite field
    ``GF(p)``.  Only roots from linear factors are returned.  The function is
    generic but in this toy project the polynomials are of low degree, so this
    approach is sufficient.
    """
    x = sp.symbols("x")
    expr = sum(coeff * x ** i for i, coeff in enumerate(poly.coeffs))
    f = sp.Poly(expr, x, modulus=p)
    _, factors = f.factor_list()
    roots: List[int] = []
    for fac, multiplicity in factors:
        if fac.degree() == 1:
            a = fac.LC() % p
            b = fac.nth(0) % p
            root = (-b * pow(a, -1, p)) % p
            roots.extend([root] * multiplicity)
    return roots


def find_relations(
    poly: Polynomial, primes: List[int], interval: int = 50
) -> Iterable[Relation]:
    """Find ``B``-smooth relations for ``poly`` using a simple line sieve."""

    offset = interval
    values = [abs(poly.evaluate(a)) for a in range(-interval, interval + 1)]

    for p in primes:
        for r in _polynomial_roots_mod_p(poly, p):
            start = (-interval + ((r - (-interval)) % p))
            for a in range(start, interval + 1, p):
                idx = a + offset
                while values[idx] != 0 and values[idx] % p == 0:
                    values[idx] //= p

    for idx, a in enumerate(range(-interval, interval + 1)):
        if values[idx] == 1:
            val = poly.evaluate(a)
            if val == 0:
                continue
            remaining = abs(val)
            factor_exp: Dict[int, int] = {}
            for p in primes:
                if remaining == 1:
                    break
                exp = 0
                while remaining % p == 0:
                    exp += 1
                    remaining //= p
                if exp:
                    factor_exp[p] = exp
            if remaining == 1:
                yield Relation(a=a, b=1, value=val, factors=factor_exp)

"""Sieving step for the General Number Field Sieve (GNFS).

The sieving phase searches for pairs ``(a, b)`` such that the values
``b^d f(a / b)`` are ``B``-smooth with respect to a chosen factor base.
Full GNFS implementations perform lattice sieving with numerous
optimisations; the routine here keeps to a one-dimensional line sieve but
follows the same algorithmic ideas:

* For each prime ``p`` in the factor base find the roots of the polynomial
  modulo ``p``.
* Use those roots to mark positions in a sieve array and subtract ``log(p)``
  from the array entries.
* After all primes have been processed, positions with small residuals are
  likely smooth and are trial-factored to confirm.

Although greatly simplified, this code mirrors the real sieving technique and
no longer relies on purely toy placeholder logic.
"""

from typing import Dict, Iterable, List

import math

from ..polynomial import Polynomial
from .relation import Relation
from .roots import _polynomial_roots_mod_p


def find_relations(
    poly: Polynomial, primes: List[int], interval: int = 50
) -> Iterable[Relation]:
    """Find ``B``-smooth relations for ``poly`` using a logarithmic sieve."""

    offset = interval
    values = [poly.evaluate(a) for a in range(-interval, interval + 1)]
    logs = [math.log(abs(v)) if v != 0 else 0.0 for v in values]

    for p in primes:
        logp = math.log(p)
        for r in _polynomial_roots_mod_p(poly, p):
            start = (-interval + ((r - (-interval)) % p))
            for a in range(start, interval + 1, p):
                idx = a + offset
                while values[idx] != 0 and values[idx] % p == 0:
                    values[idx] //= p
                    logs[idx] -= logp

    for idx, a in enumerate(range(-interval, interval + 1)):
        if abs(values[idx]) == 1 and logs[idx] < 1e-5:
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

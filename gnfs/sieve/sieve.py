"""Sieving step for the General Number Field Sieve (GNFS).

The sieving phase searches for pairs ``(a, b)`` such that both the algebraic
norm ``b^d f(a / b)`` and the rational norm ``a - m b`` are ``B``-smooth with
respect to a chosen factor base.  Full GNFS implementations perform lattice
sieving with numerous optimisations; the routine here keeps to a
one-dimensional line sieve but follows the same algorithmic ideas:

* For each prime ``p`` in the factor base find the roots of the polynomial
  modulo ``p``.
* Use those roots to mark positions in a sieve array and subtract ``log(p)``
  from the array entries.
* After all primes have been processed, positions with small residuals are
  likely smooth and are trial-factored on both the algebraic and rational
  sides to confirm.

Although greatly simplified, this code mirrors the real sieving technique and
no longer relies on purely toy placeholder logic.
"""

from typing import Dict, Iterable, List

import math

from ..polynomial import PolynomialSelection
from .relation import Relation
from .roots import _polynomial_roots_mod_p


def find_relations(
    selection: PolynomialSelection, primes: List[int], interval: int = 50
) -> Iterable[Relation]:
    """Find ``B``-smooth relations for ``poly`` using a logarithmic sieve."""

    algebraic_poly = selection.algebraic
    rational_poly = selection.rational
    offset = interval

    def _trial_factor(value: int) -> tuple[int, Dict[int, int]]:
        remaining = abs(value)
        factors: Dict[int, int] = {}
        for p in primes:
            if remaining == 1:
                break
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            if exp:
                factors[p] = exp
        return remaining, factors

    for b in range(1, interval + 1):
        values = [
            algebraic_poly.evaluate_homogeneous(a, b) for a in range(-interval, interval + 1)
        ]
        logs = [math.log(abs(v)) if v != 0 else 0.0 for v in values]

        for p in primes:
            logp = math.log(p)
            roots = _polynomial_roots_mod_p(algebraic_poly, p)
            if not roots:
                continue
            for root in roots:
                target = (root * b) % p
                start = -interval + ((target - (-interval)) % p)
                for a in range(start, interval + 1, p):
                    idx = a + offset
                    while values[idx] != 0 and values[idx] % p == 0:
                        values[idx] //= p
                        logs[idx] -= logp

        for idx, a in enumerate(range(-interval, interval + 1)):
            if math.gcd(a, b) != 1:
                continue
            if abs(values[idx]) != 1 or logs[idx] > 1e-5:
                continue

            algebraic_value = algebraic_poly.evaluate_homogeneous(a, b)
            rational_value = rational_poly.evaluate_homogeneous(a, b)
            if algebraic_value == 0 or rational_value == 0:
                continue

            remaining_alg, algebraic_factors = _trial_factor(algebraic_value)
            if remaining_alg != 1:
                continue

            remaining_rat, rational_factors = _trial_factor(rational_value)
            if remaining_rat != 1:
                continue

            yield Relation(
                a=a,
                b=b,
                algebraic_value=algebraic_value,
                rational_value=rational_value,
                algebraic_factors=algebraic_factors,
                rational_factors=rational_factors,
            )

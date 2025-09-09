"""Polynomial root utilities for sieving."""

from typing import List

import sympy as sp

from ..polynomial import Polynomial


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

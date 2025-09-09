import sympy as sp
from gnfs.polynomial.polynomial import Polynomial
from gnfs.polynomial.selection import select_polynomial
from gnfs.sieve.roots import _polynomial_roots_mod_p
from gnfs.sieve.sieve import find_relations


def test_polynomial_roots_mod_p():
    # x^2 - 1 mod 3 has roots 1 and 2
    poly = Polynomial((-1, 0, 1))
    roots = _polynomial_roots_mod_p(poly, 3)
    assert set(roots) == {1, 2}


def test_polynomial_roots_mod_p_no_root():
    poly = Polynomial((1, 0, 1))  # x^2 + 1 has no roots mod 3
    assert _polynomial_roots_mod_p(poly, 3) == []


def test_polynomial_roots_mod_p_repeated_root():
    poly = Polynomial((1, -2, 1))  # (x - 1)^2 has root 1 twice mod 5
    roots = _polynomial_roots_mod_p(poly, 5)
    assert roots == [1, 1]


def test_find_relations_produces_smooth_values():
    poly = select_polynomial(10)
    primes = list(sp.primerange(2, 6))
    relations = list(find_relations(poly, primes=primes, interval=5))
    assert relations, "Expected at least one relation"
    for rel in relations:
        value = abs(rel.value)
        for p, exp in rel.factors.items():
            for _ in range(exp):
                value //= p
        assert value == 1

import sympy as sp
from gnfs.polynomial import Polynomial, select_polynomial
from gnfs.sieve import _polynomial_roots_mod_p, find_relations


def test_polynomial_roots_mod_p():
    # x^2 - 1 mod 3 has roots 1 and 2
    poly = Polynomial((-1, 0, 1))
    roots = _polynomial_roots_mod_p(poly, 3)
    assert set(roots) == {1, 2}


def test_find_relations_produces_smooth_values():
    poly = select_polynomial(10)
    relations = list(find_relations(poly, bound=5, interval=5))
    assert relations, "Expected at least one relation"
    primes = list(sp.primerange(2, 6))
    for rel in relations:
        value = abs(rel.value)
        for p in primes:
            while value % p == 0 and value != 0:
                value //= p
        assert value == 1

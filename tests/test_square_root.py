import sympy as sp
from gnfs.polynomial import select_polynomial
from gnfs.sieve import Relation, find_relations
from gnfs.sqrt import find_factors


def test_find_factors_even_number():
    poly = select_polynomial(10)
    primes = list(sp.primerange(2, 6))
    relations = list(find_relations(poly, primes=primes, interval=5))
    factors = list(find_factors(10, relations, primes))
    assert factors == [2, 5]

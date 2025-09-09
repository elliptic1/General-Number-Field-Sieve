import sympy as sp
from gnfs.polynomial.selection import select_polynomial
from gnfs.sieve.relation import Relation
from gnfs.sieve.sieve import find_relations
from gnfs.sqrt.square_root import find_factors


def test_find_factors_even_number():
    poly = select_polynomial(10)
    primes = list(sp.primerange(2, 6))
    relations = list(find_relations(poly, primes=primes, interval=5))
    factors = list(find_factors(10, relations, primes))
    assert factors == [2, 5]


def test_find_factors_no_dependencies():
    n = 10
    relations = [Relation(a=1, b=1, value=2, factors={2: 1})]
    factors = list(find_factors(n, relations, [2]))
    assert factors == []

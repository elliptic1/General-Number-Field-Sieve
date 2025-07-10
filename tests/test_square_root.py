from gnfs.square_root import find_factors
from gnfs.sieve import Relation


def test_find_factors_even_number():
    relations = [Relation(a=0, b=1, value=1)]
    factors = list(find_factors(10, relations))
    assert factors == [2, 5]

from gnfs.linear_algebra import solve_matrix
from gnfs.sieve import Relation


def test_solve_matrix_returns_dependency():
    primes = [2]
    relations = [
        Relation(a=1, b=1, value=2, factors={2: 1}),
        Relation(a=2, b=1, value=2, factors={2: 1}),
    ]
    deps = solve_matrix(relations, primes)
    assert deps == [[0, 1]]

import numpy as np
from gnfs.linalg import solve_matrix, _nullspace_mod2
from gnfs.sieve import Relation


def test_solve_matrix_returns_dependency():
    primes = [2]
    relations = [
        Relation(a=1, b=1, value=2, factors={2: 1}),
        Relation(a=2, b=1, value=2, factors={2: 1}),
    ]
    deps = solve_matrix(relations, primes)
    assert deps == [[0, 1]]


def test_nullspace_mod2_basic_case():
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    basis = _nullspace_mod2(matrix)
    assert len(basis) == 1
    assert np.array_equal(basis[0], np.array([1, 1, 1]))

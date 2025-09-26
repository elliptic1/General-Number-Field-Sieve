import numpy as np
import numpy as np

from gnfs.linalg.matrix import _nullspace_mod2, solve_matrix
from gnfs.sieve.relation import Relation


def test_solve_matrix_returns_dependency():
    primes = [2]
    relations = [
        Relation(
            a=1,
            b=1,
            algebraic_value=2,
            rational_value=1,
            algebraic_factors={2: 1},
            rational_factors={},
        ),
        Relation(
            a=2,
            b=1,
            algebraic_value=2,
            rational_value=1,
            algebraic_factors={2: 1},
            rational_factors={},
        ),
    ]
    deps = solve_matrix(relations, primes)
    assert deps == [[0, 1]]


def test_nullspace_mod2_basic_case():
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
    basis = _nullspace_mod2(matrix)
    assert len(basis) == 1
    assert np.array_equal(basis[0], np.array([1, 1, 1]))


def test_nullspace_mod2_no_nullspace():
    matrix = np.eye(2, dtype=int)
    basis = _nullspace_mod2(matrix)
    assert basis == []


def test_solve_matrix_no_relations():
    assert solve_matrix([], [2, 3]) == []


def test_solve_matrix_multiple_primes_dependency():
    primes = [2, 3]
    relations = [
        Relation(
            a=1,
            b=1,
            algebraic_value=2,
            rational_value=1,
            algebraic_factors={2: 1},
            rational_factors={},
        ),
        Relation(
            a=1,
            b=1,
            algebraic_value=3,
            rational_value=1,
            algebraic_factors={3: 1},
            rational_factors={},
        ),
        Relation(
            a=1,
            b=1,
            algebraic_value=6,
            rational_value=1,
            algebraic_factors={2: 1, 3: 1},
            rational_factors={},
        ),
    ]
    deps = solve_matrix(relations, primes)
    assert deps == [[0, 1, 2]]

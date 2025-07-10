from gnfs.linear_algebra import solve_matrix
from gnfs.sieve import Relation


def test_solve_matrix_returns_indices():
    relations = [Relation(a=i, b=1, value=i) for i in range(3)]
    result = solve_matrix(relations)
    assert result == [0, 1, 2]

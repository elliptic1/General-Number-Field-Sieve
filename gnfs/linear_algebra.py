"""Linear algebra step for GNFS."""

from typing import Iterable, List

from .sieve import Relation


def solve_matrix(relations: Iterable[Relation]) -> List[int]:
    """Return dummy dependencies between relations."""
    # In GNFS this would build a sparse matrix and solve for nullspace.
    # Here we just return the indices of relations for demonstration.
    return list(range(len(list(relations))))

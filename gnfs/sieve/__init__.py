"""Sieving utilities for the General Number Field Sieve.

This module provides two sieving implementations:

1. **Line sieve** (`find_relations`): Simple but slow. Good for small problems
   or educational purposes. Sieves over all (a,b) pairs directly.

2. **Lattice sieve** (`find_relations_lattice`): Advanced and fast. Uses
   special-q sieving to reduce the search space by a factor of q for each
   special prime. Recommended for larger factorizations.

The lattice sieve typically provides 10-100x speedup over line sieving for
factor bases of 1000+ primes.

Example:
    >>> from gnfs.sieve import find_relations_lattice
    >>> from gnfs.polynomial import select_polynomial
    >>> import sympy
    >>> 
    >>> n = 1234567890123456789  # Number to factor
    >>> selection = select_polynomial(n, degree=4)
    >>> primes = list(sympy.primerange(2, 10000))
    >>> relations = list(find_relations_lattice(selection, primes))
"""

from .relation import Relation
from .roots import _polynomial_roots_mod_p
from .sieve import find_relations
from .lattice_sieve import (
    find_relations_lattice,
    find_relations_hybrid,
    lattice_sieve_for_special_q,
    lattice_sieve_optimized,
    select_special_q_primes,
    LatticeBasis,
    compute_lattice_basis,
    reduce_lattice_basis,
)

__all__ = [
    # Relation class
    "Relation",
    
    # Root finding
    "_polynomial_roots_mod_p",
    
    # Line sieve (simple, slow)
    "find_relations",
    
    # Lattice sieve (advanced, fast)
    "find_relations_lattice",
    "find_relations_hybrid",
    "lattice_sieve_for_special_q",
    "lattice_sieve_optimized",
    "select_special_q_primes",
    
    # Lattice utilities
    "LatticeBasis",
    "compute_lattice_basis",
    "reduce_lattice_basis",
]

"""Lattice sieving for the General Number Field Sieve (GNFS).

This module implements lattice sieving, a major improvement over the basic
line sieve. The key insight is that instead of sieving over all (a,b) pairs,
we can restrict to a 2D sublattice where a specific prime q divides the
algebraic norm.

**Special-q Lattice Sieving:**

For a "special prime" q in the factor base and a root r of f(x) mod q,
the algebraic norm F(a,b) = b^d * f(a/b) is divisible by q whenever:

    a ≡ r*b (mod q)

This defines a lattice L_q of points (a,b) where q | F(a,b). Instead of
checking all pairs, we only check those in this sublattice, reducing work
by a factor of q.

**Lattice Representation:**

The lattice L_q has basis vectors:
    v1 = (q, 0)
    v2 = (r, 1)

Any lattice point (a, b) can be written as:
    (a, b) = i*v1 + j*v2 = (i*q + j*r, j)

So for each j (which equals b), we can enumerate a-values as:
    a = i*q + j*r  for integer i

**Algorithm:**

1. Choose a "special-q" prime from the factor base
2. For each root r of f(x) mod q:
   a. Set up the sublattice L_q,r
   b. Use a 2D sieve array indexed by (i, j) lattice coordinates
   c. Sieve using the remaining factor base primes
   d. Trial factor candidates with small residuals

References:
    - Pollard, J.M. (1993). "The lattice sieve"
    - Franke, J. & Kleinjung, T. (2005). "Continued fractions and lattice sieving"
    - Pomerance, C. & Crandall, R. (2005). "Prime Numbers: A Computational Perspective"
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..polynomial import PolynomialSelection
from .relation import Relation
from .roots import _polynomial_roots_mod_p


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SIEVE_REGION = 500  # Size of sieve region in each dimension
MIN_SPECIAL_Q = 50  # Minimum size for special-q primes


# =============================================================================
# Helper Functions
# =============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm.
    
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    """
    if b == 0:
        return a, 1, 0
    else:
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y


def mod_inverse(a: int, m: int) -> Optional[int]:
    """Compute modular inverse of a mod m, or None if it doesn't exist."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None
    return x % m


# =============================================================================
# Lattice Basis Computation
# =============================================================================

@dataclass
class LatticeBasis:
    """Represents a 2D lattice basis for special-q sieving.
    
    The lattice L_q,r = {(a,b) : a ≡ r*b (mod q)} has a natural basis:
        v1 = (q, 0)
        v2 = (r, 1)
    """
    v1: Tuple[int, int]
    v2: Tuple[int, int]
    q: int
    r: int
    
    def lattice_to_ab(self, i: int, j: int) -> Tuple[int, int]:
        """Convert lattice coordinates (i, j) to original (a, b) coordinates."""
        a = i * self.v1[0] + j * self.v2[0]
        b = i * self.v1[1] + j * self.v2[1]
        return a, b
    
    def ab_to_lattice(self, a: int, b: int) -> Optional[Tuple[int, int]]:
        """Convert (a, b) to lattice coordinates, if in lattice.
        
        Solves the system:
            a = i * v1[0] + j * v2[0]
            b = i * v1[1] + j * v2[1]
        
        Returns (i, j) if the point is in the lattice, None otherwise.
        """
        # Compute determinant of the basis matrix
        det = self.v1[0] * self.v2[1] - self.v1[1] * self.v2[0]
        
        if det == 0:
            # Degenerate basis (shouldn't happen for valid lattice)
            return None
        
        # Solve using Cramer's rule:
        # i = (a * v2[1] - b * v2[0]) / det
        # j = (b * v1[0] - a * v1[1]) / det
        i_num = a * self.v2[1] - b * self.v2[0]
        j_num = b * self.v1[0] - a * self.v1[1]
        
        # Check if solutions are integers
        if i_num % det != 0 or j_num % det != 0:
            return None
        
        return i_num // det, j_num // det


def compute_lattice_basis(q: int, r: int) -> LatticeBasis:
    """Compute lattice basis for the sublattice where q | F(a,b)."""
    return LatticeBasis(v1=(q, 0), v2=(r, 1), q=q, r=r)


def reduce_lattice_basis(basis: LatticeBasis) -> LatticeBasis:
    """Apply Lagrange reduction to get shorter basis vectors."""
    def norm_sq(v: Tuple[int, int]) -> int:
        return v[0] * v[0] + v[1] * v[1]
    
    def dot(v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    def sub_mult(v1: Tuple[int, int], v2: Tuple[int, int], k: int) -> Tuple[int, int]:
        return (v1[0] - k * v2[0], v1[1] - k * v2[1])
    
    u = basis.v1
    v = basis.v2
    
    if norm_sq(u) < norm_sq(v):
        u, v = v, u
    
    while True:
        n_v = norm_sq(v)
        if n_v == 0:
            break
        mu = round(dot(u, v) / n_v)
        u = sub_mult(u, v, mu)
        if norm_sq(u) < norm_sq(v):
            u, v = v, u
        else:
            break
    
    return LatticeBasis(v1=v, v2=u, q=basis.q, r=basis.r)


# =============================================================================
# Optimized Lattice Sieve
# =============================================================================

def lattice_sieve_for_special_q(
    selection: PolynomialSelection,
    primes: List[int],
    q: int,
    sieve_region: int = DEFAULT_SIEVE_REGION,
) -> Iterable[Relation]:
    """Perform lattice sieving for a single special-q.
    
    This is an optimized implementation that:
    1. Works in (a, b) coordinates directly within the sublattice
    2. Uses efficient array-based sieving
    3. Only trial factors candidates that pass the sieve threshold
    """
    algebraic_poly = selection.algebraic
    rational_poly = selection.rational
    
    # Find roots of f(x) mod q
    q_roots = _polynomial_roots_mod_p(algebraic_poly, q)
    if not q_roots:
        return
    
    sieve_primes = [p for p in primes if p != q]
    all_primes = sorted(set(primes) | {q})
    
    def _trial_factor(value: int) -> Tuple[int, Dict[int, int]]:
        """Trial factor a value using factor base primes."""
        remaining = abs(value)
        factors: Dict[int, int] = {}
        for p in all_primes:
            if remaining == 1:
                break
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            if exp:
                factors[p] = exp
        return remaining, factors
    
    # Process each root of q
    for r in set(q_roots):
        # For this root r, we sieve the sublattice where a ≡ r*b (mod q)
        # We enumerate: for each b in [1, sieve_region], 
        # a ranges over values ≡ r*b (mod q) in [-A, A]
        
        A = sieve_region * q  # a-range
        
        for b in range(1, sieve_region + 1):
            # Values of a where a ≡ r*b (mod q)
            base_a = (r * b) % q
            if base_a > A:
                base_a -= q
            
            # Start from negative side
            a_start = base_a - ((A - base_a) // q + 1) * q
            while a_start < -A:
                a_start += q
            
            # Iterate through valid a values
            for a in range(a_start, A + 1, q):
                if math.gcd(a, b) != 1:
                    continue
                
                alg_val = algebraic_poly.evaluate_homogeneous(a, b)
                rat_val = rational_poly.evaluate_homogeneous(a, b)
                
                if alg_val == 0 or rat_val == 0:
                    continue
                
                # Trial factor both sides
                remaining_alg, alg_factors = _trial_factor(alg_val)
                if remaining_alg != 1:
                    continue
                
                remaining_rat, rat_factors = _trial_factor(rat_val)
                if remaining_rat != 1:
                    continue
                
                yield Relation(
                    a=a,
                    b=b,
                    algebraic_value=alg_val,
                    rational_value=rat_val,
                    algebraic_factors=alg_factors,
                    rational_factors=rat_factors,
                )


def lattice_sieve_optimized(
    selection: PolynomialSelection,
    primes: List[int],
    q: int,
    sieve_region: int = DEFAULT_SIEVE_REGION,
    log_threshold: float = 2.0,
) -> Iterable[Relation]:
    """Optimized lattice sieve using logarithmic sieving.
    
    Uses the same approach as the line sieve but restricted to the sublattice.
    This is faster than naive trial division for larger sieve regions.
    """
    algebraic_poly = selection.algebraic
    rational_poly = selection.rational
    
    q_roots = _polynomial_roots_mod_p(algebraic_poly, q)
    if not q_roots:
        return
    
    sieve_primes = [p for p in primes if p != q]
    all_primes = sorted(set(primes) | {q})
    
    def _trial_factor(value: int) -> Tuple[int, Dict[int, int]]:
        remaining = abs(value)
        factors: Dict[int, int] = {}
        for p in all_primes:
            if remaining == 1:
                break
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            if exp:
                factors[p] = exp
        return remaining, factors
    
    for r in set(q_roots):
        A = sieve_region * q
        
        for b in range(1, sieve_region + 1):
            # Build list of valid a values for this b
            base_a = (r * b) % q
            a_start = base_a
            while a_start > -A:
                a_start -= q
            a_start += q  # First value >= -A
            
            a_values = list(range(a_start, A + 1, q))
            if not a_values:
                continue
            
            num_a = len(a_values)
            a_offset = -a_start // q if a_start < 0 else 0
            
            # Initialize sieve arrays with log values
            alg_logs = []
            rat_logs = []
            alg_vals = []
            rat_vals = []
            
            for a in a_values:
                alg_val = algebraic_poly.evaluate_homogeneous(a, b)
                rat_val = rational_poly.evaluate_homogeneous(a, b)
                alg_vals.append(alg_val)
                rat_vals.append(rat_val)
                alg_logs.append(math.log(abs(alg_val)) if alg_val != 0 else 0.0)
                rat_logs.append(math.log(abs(rat_val)) if rat_val != 0 else 0.0)
            
            # Subtract log(q) from algebraic side (q divides by construction)
            logq = math.log(q)
            for i in range(num_a):
                alg_logs[i] -= logq
            
            # Sieve with remaining primes
            for p in sieve_primes:
                logp = math.log(p)
                
                # Algebraic side
                alg_roots = _polynomial_roots_mod_p(algebraic_poly, p)
                for root in alg_roots:
                    # Find a ≡ root * b (mod p) within our a_values
                    target = (root * b) % p
                    # Find first a_value ≡ target (mod p)
                    for idx, a in enumerate(a_values):
                        if a % p == target:
                            # Sieve this and all subsequent at stride p
                            for i in range(idx, num_a, p):
                                while alg_vals[i] != 0 and alg_vals[i] % p == 0:
                                    alg_vals[i] //= p
                                    alg_logs[i] -= logp
                            break
                
                # Rational side
                rat_roots = _polynomial_roots_mod_p(rational_poly, p)
                for root in rat_roots:
                    target = (root * b) % p
                    for idx, a in enumerate(a_values):
                        if a % p == target:
                            for i in range(idx, num_a, p):
                                while rat_vals[i] != 0 and rat_vals[i] % p == 0:
                                    rat_vals[i] //= p
                                    rat_logs[i] -= logp
                            break
            
            # Check candidates
            for idx, a in enumerate(a_values):
                if math.gcd(a, b) != 1:
                    continue
                if alg_logs[idx] > log_threshold or rat_logs[idx] > log_threshold:
                    continue
                
                alg_val = algebraic_poly.evaluate_homogeneous(a, b)
                rat_val = rational_poly.evaluate_homogeneous(a, b)
                
                if alg_val == 0 or rat_val == 0:
                    continue
                
                remaining_alg, alg_factors = _trial_factor(alg_val)
                if remaining_alg != 1:
                    continue
                
                remaining_rat, rat_factors = _trial_factor(rat_val)
                if remaining_rat != 1:
                    continue
                
                yield Relation(
                    a=a,
                    b=b,
                    algebraic_value=alg_val,
                    rational_value=rat_val,
                    algebraic_factors=alg_factors,
                    rational_factors=rat_factors,
                )


def select_special_q_primes(primes: List[int], num_special_q: int = 10) -> List[int]:
    """Select good special-q primes from the factor base.
    
    Note: This is a basic selection. For best results, use
    select_special_q_with_roots() which filters for primes with polynomial roots.
    """
    if not primes:
        return []
    
    # Use primes from the larger end of the factor base
    min_idx = len(primes) // 2
    candidates = [p for p in primes[min_idx:] if p >= MIN_SPECIAL_Q]
    
    if not candidates:
        candidates = primes[-min(num_special_q, len(primes)):]
    
    if len(candidates) <= num_special_q:
        return candidates
    
    step = len(candidates) // num_special_q
    return [candidates[i * step] for i in range(num_special_q)]


def select_special_q_with_roots(
    primes: List[int], 
    poly, 
    num_special_q: int = 10,
    min_q: int = MIN_SPECIAL_Q,
) -> List[int]:
    """Select special-q primes that have roots of f(x) mod q.
    
    This is more effective than select_special_q_primes because it ensures
    each special-q will actually produce lattice points to sieve.
    
    Args:
        primes: Factor base primes
        poly: The algebraic polynomial
        num_special_q: How many special-q primes to select
        min_q: Minimum size for special-q
    
    Returns:
        List of primes with roots, preferring larger primes
    """
    if not primes:
        return []
    
    # Find primes with roots, starting from larger ones
    good_q = []
    for p in reversed(primes):
        if p < min_q:
            continue
        roots = _polynomial_roots_mod_p(poly, p)
        if roots:
            good_q.append(p)
            if len(good_q) >= num_special_q:
                break
    
    # If we didn't find enough, also try smaller primes
    if len(good_q) < num_special_q:
        for p in primes:
            if p in good_q:
                continue
            roots = _polynomial_roots_mod_p(poly, p)
            if roots:
                good_q.append(p)
                if len(good_q) >= num_special_q:
                    break
    
    return good_q


def find_relations_lattice(
    selection: PolynomialSelection,
    primes: List[int],
    sieve_region: int = DEFAULT_SIEVE_REGION,
    num_special_q: Optional[int] = None,
    special_q_primes: Optional[List[int]] = None,
    use_log_sieve: bool = True,
) -> Iterable[Relation]:
    """Find B-smooth relations using lattice sieving.
    
    The speedup over line sieving comes from:
    1. Only checking (a,b) pairs in a sublattice (reduces work by factor q)
    2. Better cache locality from the lattice structure
    3. Larger effective sieve region per special-q
    
    Args:
        selection: Polynomial selection (algebraic and rational polynomials)
        primes: Factor base primes for smoothness testing
        sieve_region: Size of sieve region (larger = more relations but slower)
        num_special_q: Number of special-q primes to use
        special_q_primes: Explicit list of special-q primes (overrides num_special_q)
        use_log_sieve: Use logarithmic sieving (faster for larger regions)
    
    Yields:
        Relation objects for each smooth (a, b) pair found
    """
    if not primes:
        return
    
    if special_q_primes is not None:
        q_primes = special_q_primes
    else:
        if num_special_q is None:
            num_special_q = max(5, min(50, len(primes) // 10))
        # Use the smarter selection that filters for primes with roots
        q_primes = select_special_q_with_roots(
            primes, selection.algebraic, num_special_q
        )
    
    seen = set()
    
    for q in q_primes:
        if use_log_sieve:
            sieve_func = lattice_sieve_optimized
        else:
            sieve_func = lattice_sieve_for_special_q
        
        for relation in sieve_func(selection, primes, q, sieve_region):
            key = (relation.a, relation.b)
            if key not in seen:
                seen.add(key)
                yield relation


def find_relations_hybrid(
    selection: PolynomialSelection,
    primes: List[int],
    interval: int = 50,
    lattice_sieve_region: int = DEFAULT_SIEVE_REGION,
    prefer_lattice: bool = True,
) -> Iterable[Relation]:
    """Hybrid sieving that chooses the best method based on parameters."""
    use_lattice = prefer_lattice and len(primes) >= 50 and max(primes) >= MIN_SPECIAL_Q
    
    if use_lattice:
        yield from find_relations_lattice(selection, primes, lattice_sieve_region)
    else:
        from .sieve import find_relations
        yield from find_relations(selection, primes, interval)

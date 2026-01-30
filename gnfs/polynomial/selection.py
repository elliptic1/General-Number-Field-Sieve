"""Polynomial selection for General Number Field Sieve (GNFS).

This module implements polynomial selection strategies for GNFS, including:
- Base-m expansion: Express n in base m for balanced coefficients
- Murphy E scoring: Rate polynomial quality based on alpha, size, and roots
- Coefficient optimization: Search for polynomials with good properties

The quality of polynomial selection significantly impacts sieving efficiency.
A good polynomial can speed up the overall factorization by 10-100x.

References:
    - Kleinjung, T. (2006). "On polynomial selection for the general number field sieve"
    - Murphy, B. (1999). "Polynomial Selection for the Number Field Sieve"
    - Brent, R. (2010). "Some Integer Factorization Algorithms using Elliptic Curves"
"""

import math
from dataclasses import dataclass
from math import comb, gcd, log, sqrt
from typing import List, Optional, Tuple

from .polynomial import Polynomial


# =============================================================================
# Constants for polynomial scoring
# =============================================================================

# Small primes used for computing alpha values and smoothness
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                59, 61, 67, 71, 73, 79, 83, 89, 97]

# Default sieving bound estimate (adjustable based on number size)
DEFAULT_SIEVE_BOUND = 10**6


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class PolynomialSelection:
    """Container bundling the rational and algebraic polynomials for GNFS.
    
    Attributes:
        algebraic: The algebraic polynomial f(x) where f(m) ≡ 0 (mod n)
        rational: The rational polynomial g(x) = x - m (linear side)
        m: The common root, where f(m) ≡ g(m) ≡ 0 (mod n)
    """

    algebraic: Polynomial
    rational: Polynomial
    m: int


@dataclass
class PolynomialScore:
    """Quality metrics for a polynomial selection.
    
    Attributes:
        alpha: Average contribution from small primes (smaller is better)
        size_score: Measure of coefficient sizes (smaller is better)
        murphy_e: Combined Murphy E score (higher is better)
        root_score: Quality of roots modulo small primes
        leading_smoothness: Smoothness of leading coefficient
    """
    
    alpha: float
    size_score: float
    murphy_e: float
    root_score: float
    leading_smoothness: float


# =============================================================================
# Alpha Value Computation
# =============================================================================

def compute_alpha(poly: Polynomial, primes: Optional[List[int]] = None) -> float:
    """Compute the alpha value for a polynomial.
    
    The alpha value measures the average contribution to smoothness from small
    primes. A negative alpha is better because it means the polynomial values
    tend to be more divisible by small primes.
    
    For each prime p, we compute the average valuation of f(x) over x in Z/pZ,
    then subtract log(p)/(p-1) times the expected number of roots.
    
    Args:
        poly: The polynomial to evaluate
        primes: List of small primes to use (default: SMALL_PRIMES)
    
    Returns:
        Alpha value (more negative is better)
    
    Mathematical background:
        alpha(f) ≈ sum over primes p of:
            (average p-adic valuation of f(x) for x in Z/pZ - expected valuation)
        
        This captures how "biased" the polynomial is toward having values
        divisible by small primes.
    """
    if primes is None:
        primes = SMALL_PRIMES
    
    coeffs = poly.coeffs
    if not coeffs or all(c == 0 for c in coeffs):
        return 0.0
    
    alpha = 0.0
    
    for p in primes:
        # Count roots of f(x) mod p
        root_count = 0
        total_valuation = 0.0
        
        for x in range(p):
            fx = poly.evaluate(x) % p
            if fx == 0:
                root_count += 1
                # For roots, estimate average valuation contribution
                # (simplified: count as having valuation 1)
                total_valuation += 1
        
        # Expected contribution from each prime
        # log(p)/(p-1) is the expected contribution from random values
        expected = log(p) / (p - 1) if p > 1 else 0
        
        # Actual average contribution from roots
        actual = (root_count / p) * log(p) if p > 1 else 0
        
        # Alpha contribution from this prime
        # Negative when polynomial has more roots than expected (good)
        alpha += actual - expected * root_count / p
    
    return alpha


def compute_alpha_projective(poly: Polynomial, primes: Optional[List[int]] = None) -> float:
    """Compute projective alpha value considering roots at infinity.
    
    This is a more accurate alpha computation that accounts for the homogeneous
    form of the polynomial used in actual sieving.
    
    Args:
        poly: The polynomial to evaluate
        primes: List of small primes to use
    
    Returns:
        Projective alpha value
    """
    if primes is None:
        primes = SMALL_PRIMES
    
    alpha = compute_alpha(poly, primes)
    
    # Add contribution from roots at infinity (leading coefficient divisibility)
    leading = poly.coeffs[-1] if poly.coeffs else 1
    if leading == 0:
        return alpha
    
    for p in primes:
        if leading % p == 0:
            # Root at infinity for this prime
            alpha -= log(p) / (p - 1)
    
    return alpha


# =============================================================================
# Size Scoring
# =============================================================================

def coefficient_size_score(poly: Polynomial) -> float:
    """Compute a size score for polynomial coefficients.
    
    Smaller coefficients lead to smaller polynomial values during sieving,
    which increases the probability of smoothness.
    
    Uses the L2 norm of coefficients weighted by their contribution to
    polynomial values in the sieving region.
    
    Args:
        poly: The polynomial to score
    
    Returns:
        Size score (smaller is better)
    """
    coeffs = poly.coeffs
    if not coeffs:
        return float('inf')
    
    # Weighted L2 norm - higher degree coefficients matter more
    # because they dominate for larger values
    total = 0.0
    degree = len(coeffs) - 1
    
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        # Weight based on position (constant term weighted less)
        weight = 1.0 + 0.5 * i / max(1, degree)
        total += weight * log(abs(c) + 1)
    
    return total


def skewness(poly: Polynomial) -> float:
    """Compute optimal skewness for a polynomial.
    
    Skewness measures the optimal ratio of the sieving region dimensions.
    For polynomial f(x) = sum(c_i * x^i), the optimal skewness s satisfies:
        |c_i| * s^i ≈ constant for all i
    
    Args:
        poly: The polynomial
    
    Returns:
        Optimal skewness value (> 1 typically)
    """
    coeffs = poly.coeffs
    if len(coeffs) < 2:
        return 1.0
    
    # Find non-zero coefficients
    nonzero = [(i, abs(c)) for i, c in enumerate(coeffs) if c != 0]
    if len(nonzero) < 2:
        return 1.0
    
    # Estimate skewness from leading and constant terms
    # s ≈ (|c_0| / |c_d|)^(1/d)
    c0 = abs(coeffs[0]) if coeffs[0] != 0 else 1
    cd = abs(coeffs[-1]) if coeffs[-1] != 0 else 1
    d = len(coeffs) - 1
    
    if d == 0:
        return 1.0
    
    return (c0 / cd) ** (1.0 / d)


def size_score_with_skewness(poly: Polynomial) -> float:
    """Compute size score accounting for optimal skewness.
    
    This is a more accurate size metric that considers how the polynomial
    will actually be evaluated during sieving with optimal skew.
    
    Args:
        poly: The polynomial to score
    
    Returns:
        Skew-adjusted size score (smaller is better)
    """
    coeffs = poly.coeffs
    if not coeffs or len(coeffs) < 2:
        return coefficient_size_score(poly)
    
    s = skewness(poly)
    
    # Evaluate |c_i * s^i| for each coefficient
    max_term = 0.0
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        term = abs(c) * (s ** i)
        max_term = max(max_term, term)
    
    # Return log of max term (represents typical polynomial value)
    return log(max_term + 1)


# =============================================================================
# Root Properties
# =============================================================================

def count_roots_mod_p(poly: Polynomial, p: int) -> int:
    """Count roots of polynomial modulo p.
    
    More roots modulo small primes increases the probability that polynomial
    values are divisible by those primes, improving smoothness probability.
    
    Args:
        poly: The polynomial
        p: A prime number
    
    Returns:
        Number of roots in Z/pZ
    """
    return sum(1 for x in range(p) if poly.evaluate(x) % p == 0)


def root_score(poly: Polynomial, primes: Optional[List[int]] = None) -> float:
    """Score polynomial based on root properties.
    
    Polynomials with many roots modulo small primes produce values that are
    more likely to be smooth.
    
    Args:
        poly: The polynomial to score
        primes: Primes to check (default: first 10 small primes)
    
    Returns:
        Root score (higher is better)
    """
    if primes is None:
        primes = SMALL_PRIMES[:10]
    
    score = 0.0
    for p in primes:
        roots = count_roots_mod_p(poly, p)
        # Weight by 1/p since smaller primes matter more
        score += roots * log(p) / p
    
    return score


# =============================================================================
# Leading Coefficient Analysis  
# =============================================================================

def smoothness_score(n: int, bound: int = 1000) -> float:
    """Score a number by its smoothness (small prime factors).
    
    A smooth leading coefficient means the algebraic norm is more likely to
    be smooth, improving relation yield during sieving.
    
    Args:
        n: Number to evaluate
        bound: B-smoothness bound
    
    Returns:
        Smoothness score (higher is better, 1.0 = completely smooth)
    """
    if n == 0:
        return 0.0
    
    remaining = abs(n)
    smooth_part = 1
    
    # Trial division by small primes
    for p in SMALL_PRIMES:
        if p > bound:
            break
        while remaining % p == 0:
            remaining //= p
            smooth_part *= p
    
    if remaining == 1:
        return 1.0  # Completely smooth
    
    # Score based on fraction that is smooth
    if smooth_part == 0:
        return 0.0
    
    return log(smooth_part) / log(abs(n)) if n != 0 and abs(n) > 1 else 1.0


# =============================================================================
# Murphy E Scoring
# =============================================================================

def murphy_e_score(poly: Polynomial, n: int, 
                   sieve_bound: Optional[int] = None) -> float:
    """Compute Murphy's E score for polynomial quality.
    
    Murphy's E function combines multiple quality metrics into a single score
    that predicts the yield of smooth relations during sieving.
    
    E(f) ∝ ∫∫ ρ(log|f(x,y)|/log(B)) dx dy
    
    where ρ is Dickman's function and B is the smoothness bound.
    
    This implementation uses a simplified approximation:
    E ≈ exp(-alpha) * size_factor * root_bonus
    
    Args:
        poly: The polynomial to score
        n: The number being factored (for context)
        sieve_bound: Smoothness bound B (estimated from n if not provided)
    
    Returns:
        Murphy E score (higher is better)
    
    Reference:
        Murphy, B. (1999). "Polynomial Selection for the Number Field Sieve"
    """
    if sieve_bound is None:
        # Estimate sieve bound from n
        # Typically B ≈ exp(sqrt(log(n) * log(log(n))))
        log_n = log(n) if n > 1 else 1
        log_log_n = log(log_n) if log_n > 1 else 1
        sieve_bound = int(math.exp(0.5 * sqrt(log_n * log_log_n)))
        sieve_bound = max(sieve_bound, 1000)
    
    # Compute alpha (more negative is better)
    alpha = compute_alpha_projective(poly)
    
    # Alpha contribution: exp(-alpha) gives bonus for good alpha
    alpha_factor = math.exp(-alpha)
    
    # Size contribution: smaller size is better
    size = size_score_with_skewness(poly)
    # Convert to factor (inverse relationship)
    size_factor = 1.0 / (1.0 + size / 10.0)
    
    # Root contribution: more roots is better
    roots = root_score(poly)
    root_factor = 1.0 + roots / 5.0
    
    # Leading coefficient smoothness
    leading = poly.coeffs[-1] if poly.coeffs else 1
    smooth = smoothness_score(leading)
    smooth_factor = 0.5 + 0.5 * smooth
    
    # Combined score
    return alpha_factor * size_factor * root_factor * smooth_factor


def score_polynomial_selection(selection: PolynomialSelection, n: int) -> PolynomialScore:
    """Compute complete quality metrics for a polynomial selection.
    
    Args:
        selection: The polynomial selection to score
        n: The number being factored
    
    Returns:
        PolynomialScore with all quality metrics
    """
    poly = selection.algebraic
    
    alpha = compute_alpha_projective(poly)
    size = size_score_with_skewness(poly)
    murphy = murphy_e_score(poly, n)
    roots = root_score(poly)
    leading = poly.coeffs[-1] if poly.coeffs else 1
    smooth = smoothness_score(leading)
    
    return PolynomialScore(
        alpha=alpha,
        size_score=size,
        murphy_e=murphy,
        root_score=roots,
        leading_smoothness=smooth
    )


# =============================================================================
# Base-m Expansion
# =============================================================================

def base_m_expansion(n: int, m: int, degree: int) -> Tuple[int, ...]:
    """Express n in base m as polynomial coefficients.
    
    Given n and base m, compute coefficients (c_0, c_1, ..., c_d) such that:
        n = c_0 + c_1*m + c_2*m^2 + ... + c_d*m^d
    
    This gives a polynomial f(x) = c_0 + c_1*x + ... + c_d*x^d with f(m) = n.
    
    The coefficients may be negative (balanced representation) to minimize
    their absolute values.
    
    Args:
        n: The number to express
        m: The base
        degree: Maximum degree of the polynomial
    
    Returns:
        Tuple of coefficients (c_0, c_1, ..., c_d)
    
    Mathematical note:
        Using balanced representation where -m/2 < c_i <= m/2 gives smaller
        coefficients than the standard base-m representation.
    """
    if m <= 1:
        raise ValueError("base m must be > 1")
    
    coeffs = []
    remaining = n
    
    for _ in range(degree + 1):
        # Standard base-m digit
        digit = remaining % m
        
        # Convert to balanced representation: -m/2 < digit <= m/2
        if digit > m // 2:
            digit -= m
            remaining += m
        
        coeffs.append(digit)
        remaining //= m
    
    # Handle any overflow (shouldn't happen if m is chosen correctly)
    while remaining != 0:
        digit = remaining % m
        if digit > m // 2:
            digit -= m
            remaining += m
        coeffs.append(digit)
        remaining //= m
    
    return tuple(coeffs)


def optimal_base_m(n: int, degree: int) -> int:
    """Find optimal base m for base-m expansion of degree d.
    
    The optimal m minimizes the coefficient sizes while ensuring f(m) = n.
    For degree d, we want m ≈ n^(1/d) so that coefficients are balanced.
    
    Args:
        n: Number to factor
        degree: Polynomial degree
    
    Returns:
        Optimal base m
    
    Mathematical note:
        If m = n^(1/d), then all coefficients are O(m) = O(n^(1/d)).
        We search around this value for the best actual m.
    """
    # Start with theoretical optimum
    m_approx = int(round(n ** (1.0 / degree)))
    m_approx = max(2, m_approx)  # Ensure m > 1
    
    return m_approx


def select_base_m(n: int, degree: int) -> PolynomialSelection:
    """Select polynomial using base-m expansion.
    
    This constructs f(x) = c_0 + c_1*x + ... + c_d*x^d where the coefficients
    come from expressing n in base m, ensuring f(m) = n.
    
    Args:
        n: Number to factor
        degree: Polynomial degree
    
    Returns:
        PolynomialSelection with algebraic polynomial from base-m expansion
    """
    m = optimal_base_m(n, degree)
    
    # Get balanced base-m expansion
    coeffs = base_m_expansion(n, m, degree)
    
    # Verify f(m) = n
    algebraic_poly = Polynomial(coeffs)
    assert algebraic_poly.evaluate(m) == n, f"Base-m expansion failed: f({m}) = {algebraic_poly.evaluate(m)} != {n}"
    
    # Rational polynomial is always x - m
    rational_poly = Polynomial((-m, 1))
    
    return PolynomialSelection(algebraic=algebraic_poly, rational=rational_poly, m=m)


# =============================================================================
# Coefficient Optimization
# =============================================================================

def search_polynomial_range(n: int, degree: int, 
                            m_range: int = 100,
                            leading_range: int = 10) -> PolynomialSelection:
    """Search for good polynomial by trying multiple m values and leading coefficients.
    
    This implements a simplified version of Kleinjung's polynomial search:
    1. Try multiple values of m around the optimal m
    2. For each m, try adjusting the leading coefficient
    3. Return the polynomial with best Murphy E score
    
    Args:
        n: Number to factor
        degree: Polynomial degree
        m_range: How far to search around optimal m
        leading_range: Range for leading coefficient adjustment
    
    Returns:
        Best polynomial selection found
    """
    best_selection = None
    best_score = -float('inf')
    
    m_optimal = optimal_base_m(n, degree)
    
    # Search around optimal m
    m_min = max(2, m_optimal - m_range)
    m_max = m_optimal + m_range
    
    for m in range(m_min, m_max + 1):
        # Try base-m expansion
        try:
            coeffs = list(base_m_expansion(n, m, degree))
        except (ValueError, AssertionError):
            continue
        
        # Ensure it's the right degree
        while len(coeffs) <= degree:
            coeffs.append(0)
        coeffs = coeffs[:degree + 1]
        
        # Verify the polynomial works
        poly = Polynomial(tuple(coeffs))
        if poly.evaluate(m) != n:
            continue
        
        # Score this polynomial
        selection = PolynomialSelection(
            algebraic=poly,
            rational=Polynomial((-m, 1)),
            m=m
        )
        score = murphy_e_score(poly, n)
        
        if score > best_score:
            best_score = score
            best_selection = selection
        
        # Try small adjustments to leading coefficient
        # If we change c_d to c_d + k, we need to adjust lower coefficients
        # to maintain f(m) = n
        for k in range(-leading_range, leading_range + 1):
            if k == 0:
                continue
            
            new_coeffs = list(coeffs)
            new_coeffs[-1] += k
            
            # Adjust constant term to compensate
            # Original: f(m) = n
            # New: f'(m) = f(m) + k*m^d = n + k*m^d
            # So we need to subtract k*m^d from constant term
            adjustment = k * (m ** degree)
            new_coeffs[0] -= adjustment
            
            new_poly = Polynomial(tuple(new_coeffs))
            
            # Verify
            if new_poly.evaluate(m) != n:
                continue
            
            new_selection = PolynomialSelection(
                algebraic=new_poly,
                rational=Polynomial((-m, 1)),
                m=m
            )
            new_score = murphy_e_score(new_poly, n)
            
            if new_score > best_score:
                best_score = new_score
                best_selection = new_selection
    
    if best_selection is None:
        # Fall back to simple base-m
        return select_base_m(n, degree)
    
    return best_selection


def optimize_polynomial(selection: PolynomialSelection, n: int,
                       iterations: int = 50) -> PolynomialSelection:
    """Optimize an existing polynomial selection.
    
    Apply local optimizations to improve the Murphy E score:
    - Try small shifts in m
    - Adjust coefficient balance
    
    Args:
        selection: Initial polynomial selection
        n: Number being factored
        iterations: Maximum optimization iterations
    
    Returns:
        Optimized polynomial selection
    """
    best = selection
    best_score = murphy_e_score(selection.algebraic, n)
    
    for _ in range(iterations):
        improved = False
        
        # Try shifting m by ±1
        for dm in [-1, 1]:
            new_m = best.m + dm
            if new_m < 2:
                continue
            
            # Recompute coefficients for new m
            degree = best.algebraic.degree()
            try:
                new_coeffs = base_m_expansion(n, new_m, degree)
                new_poly = Polynomial(new_coeffs)
                
                if new_poly.evaluate(new_m) != n:
                    continue
                
                new_selection = PolynomialSelection(
                    algebraic=new_poly,
                    rational=Polynomial((-new_m, 1)),
                    m=new_m
                )
                new_score = murphy_e_score(new_poly, n)
                
                if new_score > best_score:
                    best = new_selection
                    best_score = new_score
                    improved = True
            except (ValueError, AssertionError):
                continue
        
        if not improved:
            break
    
    return best


# =============================================================================
# Main Selection Functions
# =============================================================================

def select_polynomial(n: int, degree: int = 1, 
                      optimize: bool = True) -> PolynomialSelection:
    """Construct a polynomial with a root ``m`` modulo ``n``.
    
    This function implements multiple polynomial selection strategies and
    returns the best polynomial found:
    
    1. For degree 1: Returns the classic linear polynomial
    2. For degree > 1: Uses base-m expansion with optimization
    
    The polynomial satisfies:
    - f(m) ≡ 0 (mod n), where f is the algebraic polynomial
    - g(m) = 0, where g(x) = x - m is the rational polynomial
    
    Parameters
    ----------
    n : int
        Integer to factor.
    degree : int
        Desired degree of the polynomial. Must be a positive integer.
        Typical values: 4 for ~80 digits, 5 for ~100 digits, 6 for ~130+ digits
    optimize : bool
        Whether to perform additional optimization passes.
    
    Returns
    -------
    PolynomialSelection
        Container with algebraic polynomial, rational polynomial, and root m.
    
    Examples
    --------
    >>> selection = select_polynomial(15, degree=2)
    >>> selection.algebraic.evaluate(selection.m) % 15
    0
    >>> selection.rational.evaluate(selection.m)
    0
    
    Notes
    -----
    The quality of polynomial selection significantly impacts GNFS efficiency.
    For production use on large numbers, more sophisticated selection methods
    (Kleinjung's algorithm) would be needed.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    if degree == 1:
        # For degree 1, use simple x - m where m ≈ √n
        m = int(math.isqrt(n))
        algebraic_coeffs = (-m, 1)
        algebraic_poly = Polynomial(algebraic_coeffs)
        rational_poly = Polynomial((-m, 1))
        return PolynomialSelection(algebraic=algebraic_poly, rational=rational_poly, m=m)
    
    # For higher degrees, use improved selection
    if optimize:
        # Search for good polynomial with optimization
        selection = search_polynomial_range(n, degree)
        selection = optimize_polynomial(selection, n)
    else:
        # Just use base-m expansion
        selection = select_base_m(n, degree)
    
    return selection


def select_polynomial_classic(n: int, degree: int = 1) -> PolynomialSelection:
    """Classic polynomial selection using (x + m)^d - n expansion.
    
    This is the original naive implementation, kept for comparison and
    backward compatibility. The expansion (x + m)^d - n yields coefficients
    from binomial expansion, which are often larger than optimal.
    
    Parameters
    ----------
    n : int
        Integer to factor.
    degree : int
        Desired degree of the polynomial.
    
    Returns
    -------
    PolynomialSelection
        The polynomial selection using classic method.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    if degree == 1:
        m = int(math.isqrt(n))
        algebraic_coeffs = (-m, 1)
    else:
        # Compute m ≈ n^{1/degree} and expand (x + m)^degree - n
        m = round(n ** (1 / degree))
        algebraic_coeffs = [comb(degree, k) * (m ** (degree - k)) for k in range(degree + 1)]
        algebraic_coeffs[0] -= n
        algebraic_coeffs = tuple(algebraic_coeffs)

    algebraic_poly = Polynomial(algebraic_coeffs)
    rational_poly = Polynomial((-m, 1))
    return PolynomialSelection(algebraic=algebraic_poly, rational=rational_poly, m=m)


def compare_polynomials(n: int, degree: int) -> dict:
    """Compare different polynomial selection methods.
    
    Useful for benchmarking and understanding the impact of polynomial quality.
    
    Args:
        n: Number to factor
        degree: Polynomial degree
    
    Returns:
        Dictionary with comparison metrics for each method
    """
    results = {}
    
    # Classic method
    classic = select_polynomial_classic(n, degree)
    classic_score = score_polynomial_selection(classic, n)
    results['classic'] = {
        'selection': classic,
        'score': classic_score,
        'coeffs': classic.algebraic.coeffs
    }
    
    # Base-m method
    base_m = select_base_m(n, degree)
    base_m_score = score_polynomial_selection(base_m, n)
    results['base_m'] = {
        'selection': base_m,
        'score': base_m_score,
        'coeffs': base_m.algebraic.coeffs
    }
    
    # Optimized method
    optimized = select_polynomial(n, degree, optimize=True)
    optimized_score = score_polynomial_selection(optimized, n)
    results['optimized'] = {
        'selection': optimized,
        'score': optimized_score,
        'coeffs': optimized.algebraic.coeffs
    }
    
    return results

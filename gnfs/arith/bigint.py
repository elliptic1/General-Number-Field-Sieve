"""Big integer arithmetic with gmpy2 acceleration.

Provides a consistent interface for number-theoretic operations,
using gmpy2 when available for significant speedups (10-100x for
large numbers), with pure Python fallbacks for portability.

Performance notes:
- gmpy2 uses GMP (GNU Multiple Precision) library
- For 100+ digit numbers, gmpy2 is essential for reasonable performance
- Pure Python fallback works but is much slower for large numbers

Example speedups with gmpy2:
- Integer multiplication: 10-50x faster
- GCD: 10-30x faster
- Primality testing: 50-100x faster
- Modular exponentiation: 20-50x faster
"""

import math
from typing import Optional, Tuple, List

# Try to import gmpy2, set flag for availability
try:
    import gmpy2
    from gmpy2 import mpz as _mpz
    HAVE_GMPY2 = True
except ImportError:
    gmpy2 = None
    HAVE_GMPY2 = False


# =============================================================================
# Integer Type
# =============================================================================

def mpz(x) -> int:
    """Convert to multi-precision integer.
    
    Uses gmpy2.mpz when available for faster operations,
    otherwise returns Python int.
    """
    if HAVE_GMPY2:
        return _mpz(x)
    return int(x)


# =============================================================================
# Basic Arithmetic
# =============================================================================

def isqrt(n: int) -> int:
    """Integer square root: largest x such that x² ≤ n."""
    if HAVE_GMPY2:
        return int(gmpy2.isqrt(n))
    return math.isqrt(n)


def iroot(n: int, k: int) -> Tuple[int, bool]:
    """Integer k-th root.
    
    Returns (root, exact) where root is floor(n^(1/k))
    and exact is True if root^k == n.
    """
    if HAVE_GMPY2:
        root, exact = gmpy2.iroot(n, k)
        return int(root), exact
    
    # Pure Python fallback using Newton's method
    if n < 0:
        if k % 2 == 0:
            raise ValueError("Even root of negative number")
        return (-iroot(-n, k)[0], False)
    
    if n == 0:
        return (0, True)
    if k == 1:
        return (n, True)
    if k == 2:
        root = math.isqrt(n)
        return (root, root * root == n)
    
    # Newton's method for k-th root
    x = int(n ** (1 / k)) + 1
    while True:
        x_new = ((k - 1) * x + n // (x ** (k - 1))) // k
        if x_new >= x:
            break
        x = x_new
    
    return (x, x ** k == n)


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    if HAVE_GMPY2:
        return int(gmpy2.gcd(a, b))
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    if HAVE_GMPY2:
        return int(gmpy2.lcm(a, b))
    return abs(a * b) // gcd(a, b) if a and b else 0


def mod_inverse(a: int, m: int) -> Optional[int]:
    """Modular inverse: x such that a*x ≡ 1 (mod m).
    
    Returns None if inverse doesn't exist (gcd(a, m) ≠ 1).
    """
    if HAVE_GMPY2:
        try:
            return int(gmpy2.invert(a, m))
        except ZeroDivisionError:
            return None
    
    # Extended Euclidean algorithm
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        return None
    return x % m


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm.
    
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    """
    if b == 0:
        return a, 1, 0
    g, x, y = _extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def powmod(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: base^exp mod m."""
    if HAVE_GMPY2:
        return int(gmpy2.powmod(base, exp, mod))
    return pow(base, exp, mod)


# =============================================================================
# Number Theory
# =============================================================================

def jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a/n).
    
    Generalization of Legendre symbol to odd composites.
    Returns -1, 0, or 1.
    """
    if HAVE_GMPY2:
        return gmpy2.jacobi(a, n)
    
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be positive odd integer")
    
    a = a % n
    result = 1
    
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result
        
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    
    return result if n == 1 else 0


def is_prime(n: int, rounds: int = 25) -> bool:
    """Miller-Rabin primality test.
    
    Args:
        n: Number to test
        rounds: Number of rounds (more = more certain, default 25)
    
    Returns:
        True if probably prime, False if definitely composite
    """
    if HAVE_GMPY2:
        return gmpy2.is_prime(n, rounds)
    
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    
    # Write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Deterministic witnesses for small n
    if n < 2047:
        witnesses = [2]
    elif n < 1373653:
        witnesses = [2, 3]
    elif n < 9080191:
        witnesses = [31, 73]
    elif n < 25326001:
        witnesses = [2, 3, 5]
    elif n < 3215031751:
        witnesses = [2, 3, 5, 7]
    elif n < 4759123141:
        witnesses = [2, 7, 61]
    elif n < 1122004669633:
        witnesses = [2, 13, 23, 1662803]
    elif n < 2152302898747:
        witnesses = [2, 3, 5, 7, 11]
    elif n < 3474749660383:
        witnesses = [2, 3, 5, 7, 11, 13]
    elif n < 341550071728321:
        witnesses = [2, 3, 5, 7, 11, 13, 17]
    else:
        # For very large n, use random witnesses
        import random
        witnesses = [random.randrange(2, n - 1) for _ in range(rounds)]
    
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def next_prime(n: int) -> int:
    """Find smallest prime > n."""
    if HAVE_GMPY2:
        return int(gmpy2.next_prime(n))
    
    if n < 2:
        return 2
    
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(candidate):
        candidate += 2
    return candidate


def prev_prime(n: int) -> int:
    """Find largest prime < n."""
    if n <= 2:
        raise ValueError("No prime less than 2")
    if n == 3:
        return 2
    
    candidate = n - 1 if n % 2 == 0 else n - 2
    while candidate > 1 and not is_prime(candidate):
        candidate -= 2
    
    if candidate < 2:
        return 2
    return candidate


def is_power(n: int) -> Tuple[bool, int, int]:
    """Check if n is a perfect power.
    
    Returns (is_power, base, exp) where n = base^exp if is_power,
    or (False, n, 1) otherwise.
    """
    if HAVE_GMPY2:
        result = gmpy2.is_power(n)
        if result:
            # Find the actual base and exponent
            for k in range(2, n.bit_length() + 1):
                root, exact = iroot(n, k)
                if exact:
                    # Check if root is also a power
                    inner_power, inner_base, inner_exp = is_power(root)
                    if inner_power:
                        return True, inner_base, k * inner_exp
                    return True, root, k
        return False, n, 1
    
    # Pure Python
    if n <= 1:
        return (n == 1, 1, 1) if n == 1 else (False, n, 1)
    
    for k in range(2, n.bit_length() + 1):
        root, exact = iroot(n, k)
        if exact and root > 1:
            # Check if root is also a power
            inner_power, inner_base, inner_exp = is_power(root)
            if inner_power:
                return True, inner_base, k * inner_exp
            return True, root, k
    
    return False, n, 1


# =============================================================================
# Factorization Utilities
# =============================================================================

def factor_trial(n: int, limit: int = 10000) -> Tuple[List[Tuple[int, int]], int]:
    """Trial division factorization up to limit.
    
    Args:
        n: Number to factor
        limit: Trial division limit
    
    Returns:
        Tuple of (factors, remaining) where factors is list of (prime, exp)
        and remaining is the unfactored part.
    """
    factors = []
    
    # Handle 2 specially
    exp = 0
    while n % 2 == 0:
        n //= 2
        exp += 1
    if exp:
        factors.append((2, exp))
    
    # Odd primes up to limit
    p = 3
    while p <= limit:
        if p * p > n and n > 1:
            # n is prime, add it as a factor
            if n <= limit:
                factors.append((n, 1))
                n = 1
            break
        exp = 0
        while n % p == 0:
            n //= p
            exp += 1
        if exp:
            factors.append((p, exp))
        p += 2
    
    return factors, n


def primes_up_to(limit: int) -> List[int]:
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    
    if HAVE_GMPY2:
        # Use gmpy2's next_prime for potentially faster generation
        primes = []
        p = 2
        while p <= limit:
            primes.append(p)
            p = int(gmpy2.next_prime(p))
        return primes
    
    # Sieve of Eratosthenes
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    
    return [i for i, is_p in enumerate(sieve) if is_p]


# =============================================================================
# Utility Functions
# =============================================================================

def bit_length(n: int) -> int:
    """Number of bits needed to represent n."""
    return n.bit_length()


def digit_count(n: int, base: int = 10) -> int:
    """Number of digits in base representation of n."""
    if n == 0:
        return 1
    n = abs(n)
    if base == 10:
        return len(str(n))
    count = 0
    while n:
        n //= base
        count += 1
    return count

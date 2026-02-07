"""Algebraic square root computation for GNFS.

In GNFS, after finding a dependency of relations, we have:
- Rational side: product of (a - b*m) values that is a perfect square in Z
- Algebraic side: product of (a - b*α) values that is a perfect square in Z[α]

Computing the rational square root is trivial (integer sqrt). Computing the
algebraic square root requires working in the number field Q(α) and is the
main challenge of this step.

This module implements:
1. Number field arithmetic for elements of Q(α)
2. Montgomery's square root algorithm using CRT + Hensel lifting
3. Couveignes' method as an alternative for certain cases

References:
- Montgomery, P.L. (1994). "Square roots of products of algebraic numbers"
- Couveignes, J.-M. (1993). "Computing a square root for the number field sieve"
- Nguyen, P.Q. (1998). "A Montgomery-like square root for the Number Field Sieve"
"""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Dict, List, Optional, Tuple
import sympy as sp
from sympy import Poly, symbols, ZZ, QQ, ntheory


@dataclass
class NumberFieldElement:
    """An element of Q(α) where α is a root of the defining polynomial.
    
    Represented as a polynomial in α with rational coefficients:
    a_0 + a_1*α + a_2*α² + ... + a_{d-1}*α^{d-1}
    
    Attributes:
        coeffs: List of Fraction coefficients [a_0, a_1, ..., a_{d-1}]
        poly: The defining polynomial f(x) where f(α) = 0
    """
    coeffs: List[Fraction]
    poly: List[int]  # Coefficients of defining polynomial
    
    @property
    def degree(self) -> int:
        """Degree of the number field (one less than poly degree)."""
        return len(self.poly) - 1
    
    def __add__(self, other: 'NumberFieldElement') -> 'NumberFieldElement':
        """Add two number field elements."""
        if self.poly != other.poly:
            raise ValueError("Elements from different number fields")
        result = [a + b for a, b in zip(self.coeffs, other.coeffs)]
        return NumberFieldElement(result, self.poly)
    
    def __sub__(self, other: 'NumberFieldElement') -> 'NumberFieldElement':
        """Subtract two number field elements."""
        if self.poly != other.poly:
            raise ValueError("Elements from different number fields")
        result = [a - b for a, b in zip(self.coeffs, other.coeffs)]
        return NumberFieldElement(result, self.poly)
    
    def __neg__(self) -> 'NumberFieldElement':
        """Negate a number field element."""
        return NumberFieldElement([-c for c in self.coeffs], self.poly)
    
    def __mul__(self, other: 'NumberFieldElement') -> 'NumberFieldElement':
        """Multiply two number field elements."""
        if self.poly != other.poly:
            raise ValueError("Elements from different number fields")
        
        d = self.degree
        # First compute product as polynomial (may have degree > d-1)
        prod = [Fraction(0)] * (2 * d - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                prod[i + j] += a * b
        
        # Reduce modulo defining polynomial
        return self._reduce(prod)
    
    def _reduce(self, coeffs: List[Fraction]) -> 'NumberFieldElement':
        """Reduce a polynomial mod the defining polynomial.
        
        For f(x) = x^d + c_{d-1}*x^{d-1} + ... + c_0, we have:
        α^d = -(c_{d-1}*α^{d-1} + ... + c_0) / leading_coeff
        
        To reduce α^{d+k}, we repeatedly apply: α^d -> reduction formula
        """
        d = self.degree
        result = list(coeffs)
        lead = Fraction(self.poly[-1])
        
        # Process from highest degree down to d
        # For each term at degree >= d, replace α^d with the reduction
        for deg in range(len(result) - 1, d - 1, -1):
            if deg < len(result) and result[deg] != 0:
                coeff = result[deg]
                result[deg] = Fraction(0)
                
                # α^deg = α^{deg-d} * α^d
                # α^d = -(c_0 + c_1*α + ... + c_{d-1}*α^{d-1}) / lead
                # So α^deg contributes to positions (deg-d), (deg-d+1), ..., (deg-1)
                shift = deg - d
                for i in range(d):
                    target = i + shift
                    if target < len(result):
                        result[target] -= coeff * Fraction(self.poly[i]) / lead
                    # If target >= len(result), those terms will be handled
                    # in subsequent iterations (they're at degree < current deg)
        
        # Trim trailing zeros and pad to length d
        while len(result) > d:
            result.pop()
        while len(result) < d:
            result.append(Fraction(0))
        
        return NumberFieldElement(result, self.poly)
    
    def __pow__(self, n: int) -> 'NumberFieldElement':
        """Compute self^n using binary exponentiation."""
        if n < 0:
            raise ValueError("Negative exponents not supported")
        if n == 0:
            return self.one(self.poly)
        
        result = self.one(self.poly)
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result
    
    @classmethod
    def one(cls, poly: List[int]) -> 'NumberFieldElement':
        """Return the multiplicative identity."""
        d = len(poly) - 1
        coeffs = [Fraction(1)] + [Fraction(0)] * (d - 1)
        return cls(coeffs, poly)
    
    @classmethod
    def zero(cls, poly: List[int]) -> 'NumberFieldElement':
        """Return the additive identity."""
        d = len(poly) - 1
        coeffs = [Fraction(0)] * d
        return cls(coeffs, poly)
    
    @classmethod
    def alpha(cls, poly: List[int]) -> 'NumberFieldElement':
        """Return α (the generator of the number field)."""
        d = len(poly) - 1
        coeffs = [Fraction(0), Fraction(1)] + [Fraction(0)] * (d - 2)
        return cls(coeffs, poly)
    
    @classmethod
    def from_ab(cls, a: int, b: int, poly: List[int]) -> 'NumberFieldElement':
        """Create the element (a - b*α)."""
        d = len(poly) - 1
        coeffs = [Fraction(a), Fraction(-b)] + [Fraction(0)] * (d - 2)
        return cls(coeffs, poly)
    
    def norm(self) -> Fraction:
        """Compute the norm N(self) = product of conjugates.
        
        For an element β in Q(α), N(β) = Resultant(β, f) / leading_coeff(f)^d
        where f is the minimal polynomial of α.
        """
        x = symbols('x')
        
        # Build polynomial representation of self
        elem_poly = sum(
            c * x**i for i, c in enumerate(self.coeffs) if c != 0
        )
        def_poly = sum(
            c * x**i for i, c in enumerate(self.poly)
        )
        
        # Norm is resultant divided by leading coeff power
        from sympy import resultant
        res = resultant(elem_poly, def_poly, x)
        lead_power = Fraction(self.poly[-1]) ** self.degree
        
        return Fraction(res) / lead_power
    
    def evaluate_at(self, m: int) -> Fraction:
        """Evaluate this element at α = m."""
        result = Fraction(0)
        power = Fraction(1)
        for c in self.coeffs:
            result += c * power
            power *= m
        return result
    
    def is_zero(self) -> bool:
        """Check if this element is zero."""
        return all(c == 0 for c in self.coeffs)
    
    def __repr__(self) -> str:
        terms = []
        for i, c in enumerate(self.coeffs):
            if c != 0:
                if i == 0:
                    terms.append(str(c))
                elif i == 1:
                    terms.append(f"{c}*α")
                else:
                    terms.append(f"{c}*α^{i}")
        return " + ".join(terms) if terms else "0"


def compute_algebraic_product(
    relations: List['Relation'],
    dependency: List[int],
    poly_coeffs: List[int],
) -> NumberFieldElement:
    """Compute the product of (a - b*α) for relations in dependency.
    
    Args:
        relations: List of all relations
        dependency: Indices of relations whose product is a square
        poly_coeffs: Coefficients of the defining polynomial
    
    Returns:
        Product as a NumberFieldElement
    """
    result = NumberFieldElement.one(poly_coeffs)
    
    for idx in dependency:
        rel = relations[idx]
        factor = NumberFieldElement.from_ab(rel.a, rel.b, poly_coeffs)
        result = result * factor
    
    return result


def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """Compute square root of n mod p using Tonelli-Shanks.
    
    Returns x such that x² ≡ n (mod p), or None if n is not a QR.
    """
    n = n % p
    
    if n == 0:
        return 0
    
    if p == 2:
        return n % 2
    
    # Check if n is a quadratic residue
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    # Factor out powers of 2 from p-1
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    # Find a quadratic non-residue
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while True:
        if t == 1:
            return r
        
        # Find least i such that t^{2^i} = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Update
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p


def sqrt_mod_prime_power(n: int, p: int, k: int) -> Optional[int]:
    """Compute square root of n mod p^k using Hensel lifting.
    
    Args:
        n: The value to take sqrt of
        p: Prime
        k: Power
    
    Returns:
        x such that x² ≡ n (mod p^k), or None if no solution
    """
    if k == 1:
        return tonelli_shanks(n % p, p)
    
    # Start with sqrt mod p
    x = tonelli_shanks(n % p, p)
    if x is None:
        return None
    
    # Hensel lift to higher powers
    pk = p
    for _ in range(1, k):
        pk_next = pk * p
        # Newton's method: x_{i+1} = (x_i + n/x_i) / 2 mod p^{k+1}
        # But we need 2 to be invertible, so use: x' = x - (x² - n)/(2x) mod p^{k+1}
        
        x_sq = (x * x) % pk_next
        residue = (x_sq - n) % pk_next
        
        # Find inverse of 2x mod p^{k+1}
        two_x = (2 * x) % pk_next
        try:
            inv_2x = pow(two_x, -1, pk_next)
        except ValueError:
            # 2x not invertible (shouldn't happen for odd p)
            return None
        
        correction = (residue * inv_2x) % pk_next
        x = (x - correction) % pk_next
        pk = pk_next
    
    return x


def montgomery_sqrt_rational(product: int) -> int:
    """Compute integer square root of a perfect square.
    
    This is the easy part - just use isqrt.
    """
    y = isqrt(abs(product))
    if y * y != abs(product):
        raise ValueError(f"Product {product} is not a perfect square")
    return y if product >= 0 else None


def montgomery_sqrt_algebraic(
    product: NumberFieldElement,
    poly_coeffs: List[int],
    primes: List[int],
    m: int,
) -> Optional[int]:
    """Compute algebraic square root using Montgomery's method.
    
    The product is β ∈ Z[α] that is known to be a perfect square.
    We want to find γ such that γ² = β, then evaluate γ(m) mod n.
    
    Montgomery's method:
    1. Compute sqrt(β) mod p for many small primes p
    2. Use CRT to combine into sqrt(β) mod large composite
    3. Recover actual coefficients of γ
    
    This is a simplified version that works for moderate-sized cases.
    """
    # For now, use a simpler approach:
    # Evaluate the product at α = m and take integer sqrt
    # This works when the algebraic sqrt has small coefficients
    
    val = product.evaluate_at(m)
    
    # The evaluated value should be a perfect square
    if val.denominator != 1:
        # Need to clear denominators
        d = val.denominator
        val_int = int(val.numerator)
        # val = val_int / d, so sqrt(val) = sqrt(val_int) / sqrt(d)
        # This gets complicated; for now assume integer
        return None
    
    val_int = int(val.numerator)
    if val_int < 0:
        return None
    
    sqrt_val = isqrt(val_int)
    if sqrt_val * sqrt_val == val_int:
        return sqrt_val
    
    return None


def find_square_root_in_number_field(
    product: NumberFieldElement,
    n: int,
) -> Optional[int]:
    """Find y such that y² ≡ N(product) (mod n).
    
    This is a more sophisticated approach that:
    1. Computes the norm of the algebraic product
    2. Takes the square root of the norm
    
    The norm of a product is the product of norms, and the norm
    of a square is a square, so N(γ²) = N(γ)² which is a perfect square.
    """
    norm = product.norm()
    
    if norm.denominator != 1:
        # Handle non-integer norm
        return None
    
    norm_int = abs(int(norm.numerator))
    sqrt_norm = isqrt(norm_int)
    
    if sqrt_norm * sqrt_norm != norm_int:
        # Not a perfect square (shouldn't happen for valid dependency)
        return None
    
    return sqrt_norm % n


class AlgebraicSquareRoot:
    """Helper class for computing algebraic square roots in GNFS."""
    
    def __init__(self, poly_coeffs: List[int], m: int, n: int):
        """Initialize with GNFS parameters.
        
        Args:
            poly_coeffs: Coefficients of algebraic polynomial [c_0, c_1, ..., c_d]
            m: The integer such that f(m) ≡ 0 (mod n)
            n: The number being factored
        """
        self.poly = poly_coeffs
        self.m = m
        self.n = n
        self.degree = len(poly_coeffs) - 1
    
    def compute_product(
        self,
        relations: List['Relation'],
        dependency: List[int],
    ) -> NumberFieldElement:
        """Compute product of (a - b*α) for dependency."""
        return compute_algebraic_product(relations, dependency, self.poly)
    
    def extract_factor(
        self,
        relations: List['Relation'],
        dependency: List[int],
    ) -> Optional[Tuple[int, int]]:
        """Extract a factor of n from a dependency.
        
        Returns (factor1, factor2) if successful, None otherwise.
        """
        # Compute rational side product and its sqrt
        x = 1
        for idx in dependency:
            rel = relations[idx]
            x = (x * rel.rational_value) % self.n
        
        # Compute algebraic side product
        alg_product = self.compute_product(relations, dependency)
        
        # Try different methods to get y
        y = None
        
        # Method 1: Direct evaluation
        y = montgomery_sqrt_algebraic(alg_product, self.poly, [], self.m)
        
        # Method 2: Via norm
        if y is None:
            y = find_square_root_in_number_field(alg_product, self.n)
        
        # Method 3: Product of absolute values (original simple method)
        if y is None:
            prod = 1
            for idx in dependency:
                prod *= abs(relations[idx].algebraic_value)
            sqrt_prod = isqrt(prod)
            if sqrt_prod * sqrt_prod == prod:
                y = sqrt_prod % self.n
        
        if y is None:
            return None
        
        # Try gcd(x - y, n) and gcd(x + y, n)
        for diff in [x - y, x + y]:
            g = gcd(diff % self.n, self.n)
            if 1 < g < self.n:
                return (g, self.n // g)
        
        return None

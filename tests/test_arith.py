"""Tests for big integer arithmetic module.

Tests both gmpy2 (if available) and pure Python implementations.
"""

import pytest
import math

from gnfs.arith import (
    HAVE_GMPY2,
    mpz,
    isqrt,
    gcd,
    lcm,
    mod_inverse,
    jacobi,
    is_prime,
    next_prime,
    prev_prime,
    is_power,
    iroot,
    factor_trial,
    powmod,
)
from gnfs.arith.bigint import primes_up_to, digit_count, bit_length


class TestMPZ:
    """Tests for multi-precision integer type."""
    
    def test_small_integer(self):
        """Convert small integer."""
        x = mpz(42)
        assert x == 42
    
    def test_large_integer(self):
        """Convert large integer."""
        x = mpz(10**100)
        assert x == 10**100
    
    def test_from_string(self):
        """Convert from string."""
        x = mpz("123456789012345678901234567890")
        assert x == 123456789012345678901234567890
    
    def test_arithmetic(self):
        """Basic arithmetic works."""
        a = mpz(10**50)
        b = mpz(10**25)
        
        assert a + b == 10**50 + 10**25
        assert a - b == 10**50 - 10**25
        assert a * b == 10**75
        assert a // b == 10**25


class TestIsqrt:
    """Tests for integer square root."""
    
    def test_perfect_squares(self):
        """Test with perfect squares."""
        for n in [0, 1, 4, 9, 16, 25, 100, 10000]:
            assert isqrt(n) ** 2 == n
    
    def test_non_perfect_squares(self):
        """Test with non-perfect squares."""
        for n in [2, 3, 5, 7, 10, 99]:
            root = isqrt(n)
            assert root ** 2 <= n < (root + 1) ** 2
    
    def test_large_number(self):
        """Test with large number."""
        n = 10**100
        root = isqrt(n)
        assert root == 10**50


class TestIroot:
    """Tests for integer k-th root."""
    
    def test_square_root(self):
        """k=2 is square root."""
        for n in [4, 9, 16, 25, 100]:
            root, exact = iroot(n, 2)
            assert exact
            assert root ** 2 == n
    
    def test_cube_root(self):
        """k=3 is cube root."""
        for n in [8, 27, 64, 125, 1000]:
            root, exact = iroot(n, 3)
            assert exact
            assert root ** 3 == n
    
    def test_non_exact(self):
        """Non-exact roots."""
        root, exact = iroot(10, 2)
        assert not exact
        assert root == 3  # floor(sqrt(10))
    
    def test_fourth_root(self):
        """Fourth root."""
        root, exact = iroot(16, 4)
        assert exact
        assert root == 2
    
    def test_negative_exact_cube_root(self):
        """Negative exact cube root preserves exactness (regression test)."""
        root, exact = iroot(-27, 3)
        assert root == -3, f"Expected -3, got {root}"
        assert exact == True, f"Expected exact=True for iroot(-27, 3)"
        assert root ** 3 == -27
    
    def test_negative_non_exact_cube_root(self):
        """Negative non-exact cube root."""
        root, exact = iroot(-28, 3)
        assert root == -3
        assert exact == False
        assert root ** 3 > -28 > (root - 1) ** 3
    
    def test_large_integer_no_overflow(self):
        """Large integers don't cause float overflow (regression test)."""
        # 10^1000 would overflow float, but iroot should handle it
        n = 10 ** 1000
        root, exact = iroot(n, 3)
        # Verify bounds: root^3 <= n < (root+1)^3
        assert root ** 3 <= n
        assert (root + 1) ** 3 > n
    
    def test_large_exact_power(self):
        """Large exact power."""
        base = 123456789
        n = base ** 7
        root, exact = iroot(n, 7)
        assert root == base
        assert exact == True


class TestGCD:
    """Tests for greatest common divisor."""
    
    def test_coprime(self):
        """GCD of coprime numbers is 1."""
        assert gcd(7, 11) == 1
        assert gcd(15, 28) == 1
    
    def test_common_factor(self):
        """GCD finds common factors."""
        assert gcd(12, 18) == 6
        assert gcd(100, 75) == 25
    
    def test_one_divides_other(self):
        """When one divides the other."""
        assert gcd(12, 4) == 4
        assert gcd(100, 10) == 10
    
    def test_with_zero(self):
        """GCD with zero."""
        assert gcd(5, 0) == 5
        assert gcd(0, 7) == 7
    
    def test_large_numbers(self):
        """GCD of large numbers."""
        a = 2**100 * 3**50
        b = 2**75 * 5**25
        assert gcd(a, b) == 2**75


class TestLCM:
    """Tests for least common multiple."""
    
    def test_coprime(self):
        """LCM of coprime is product."""
        assert lcm(7, 11) == 77
    
    def test_common_factor(self):
        """LCM with common factors."""
        assert lcm(12, 18) == 36
    
    def test_one_divides_other(self):
        """When one divides the other."""
        assert lcm(12, 4) == 12


class TestModInverse:
    """Tests for modular inverse."""
    
    def test_exists(self):
        """Inverse exists when coprime."""
        inv = mod_inverse(3, 7)
        assert inv is not None
        assert (3 * inv) % 7 == 1
    
    def test_not_exists(self):
        """Inverse doesn't exist when not coprime."""
        assert mod_inverse(6, 9) is None
    
    def test_various(self):
        """Test various cases."""
        for a, m in [(3, 11), (7, 13), (5, 17), (2, 31)]:
            inv = mod_inverse(a, m)
            assert inv is not None
            assert (a * inv) % m == 1


class TestJacobi:
    """Tests for Jacobi symbol."""
    
    def test_quadratic_residues(self):
        """Jacobi symbol of QR is 1."""
        # 1, 2, 4 are QRs mod 7
        assert jacobi(1, 7) == 1
        assert jacobi(2, 7) == 1
        assert jacobi(4, 7) == 1
    
    def test_non_residues(self):
        """Jacobi symbol of non-QR is -1."""
        # 3, 5, 6 are non-QRs mod 7
        assert jacobi(3, 7) == -1
        assert jacobi(5, 7) == -1
        assert jacobi(6, 7) == -1
    
    def test_zero(self):
        """Jacobi of multiple is 0."""
        assert jacobi(7, 7) == 0
        assert jacobi(14, 7) == 0


class TestIsPrime:
    """Tests for primality testing."""
    
    def test_small_primes(self):
        """Test small primes."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        for p in primes:
            assert is_prime(p), f"{p} should be prime"
    
    def test_small_composites(self):
        """Test small composites."""
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16]
        for n in composites:
            assert not is_prime(n), f"{n} should not be prime"
    
    def test_carmichael_number(self):
        """Carmichael numbers should be detected as composite."""
        # 561 = 3 * 11 * 17 is Carmichael
        assert not is_prime(561)
    
    def test_large_prime(self):
        """Test large known prime."""
        # 10th Mersenne prime: 2^89 - 1
        p = 2**89 - 1
        assert is_prime(p)
    
    def test_large_composite(self):
        """Test large composite."""
        # Product of two primes
        assert not is_prime(1000000007 * 1000000009)
    
    def test_special_cases(self):
        """Edge cases."""
        assert not is_prime(0)
        assert not is_prime(1)
        assert is_prime(2)
        assert not is_prime(-5)


class TestNextPrime:
    """Tests for next_prime."""
    
    def test_from_prime(self):
        """Next prime after a prime."""
        assert next_prime(2) == 3
        assert next_prime(3) == 5
        assert next_prime(5) == 7
        assert next_prime(7) == 11
    
    def test_from_composite(self):
        """Next prime after composite."""
        assert next_prime(4) == 5
        assert next_prime(10) == 11
        assert next_prime(100) == 101
    
    def test_from_zero(self):
        """Next prime after 0."""
        assert next_prime(0) == 2
        assert next_prime(1) == 2


class TestPrevPrime:
    """Tests for prev_prime."""
    
    def test_from_prime(self):
        """Previous prime before a prime."""
        assert prev_prime(3) == 2
        assert prev_prime(7) == 5
        assert prev_prime(11) == 7
    
    def test_from_composite(self):
        """Previous prime before composite."""
        assert prev_prime(10) == 7
        assert prev_prime(100) == 97


class TestIsPower:
    """Tests for perfect power detection."""
    
    def test_perfect_squares(self):
        """Detect perfect squares."""
        is_pow, base, exp = is_power(16)
        assert is_pow
        assert base ** exp == 16
    
    def test_perfect_cubes(self):
        """Detect perfect cubes."""
        is_pow, base, exp = is_power(27)
        assert is_pow
        assert base ** exp == 27
    
    def test_higher_powers(self):
        """Detect higher powers."""
        is_pow, base, exp = is_power(2**10)
        assert is_pow
        assert base ** exp == 2**10
    
    def test_not_power(self):
        """Non-powers detected."""
        is_pow, _, _ = is_power(10)
        assert not is_pow
    
    def test_prime(self):
        """Primes are not powers."""
        is_pow, _, _ = is_power(7)
        assert not is_pow


class TestFactorTrial:
    """Tests for trial division."""
    
    def test_small_number(self):
        """Factor small number."""
        factors, remaining = factor_trial(60, 100)
        assert remaining == 1
        # 60 = 2² × 3 × 5
        assert (2, 2) in factors
        assert (3, 1) in factors
        assert (5, 1) in factors
    
    def test_prime(self):
        """Prime number stays unfactored."""
        factors, remaining = factor_trial(97, 10)
        assert remaining == 97
        assert factors == []
    
    def test_large_factors(self):
        """Factors beyond limit stay."""
        # 101 * 103 = 10403
        factors, remaining = factor_trial(10403, 50)
        assert remaining == 10403
        assert factors == []


class TestPowmod:
    """Tests for modular exponentiation."""
    
    def test_basic(self):
        """Basic modular exponentiation."""
        assert powmod(2, 10, 1000) == 24  # 1024 mod 1000
    
    def test_large(self):
        """Large exponent."""
        # Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
        assert powmod(2, 100, 101) == 1


class TestPrimesUpTo:
    """Tests for prime generation."""
    
    def test_primes_to_20(self):
        """Primes up to 20."""
        assert primes_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    
    def test_primes_to_100(self):
        """Count primes up to 100."""
        primes = primes_up_to(100)
        assert len(primes) == 25  # π(100) = 25
    
    def test_empty(self):
        """No primes below 2."""
        assert primes_up_to(1) == []


class TestUtilities:
    """Tests for utility functions."""
    
    def test_bit_length(self):
        """Test bit length."""
        assert bit_length(0) == 0
        assert bit_length(1) == 1
        assert bit_length(7) == 3
        assert bit_length(8) == 4
        assert bit_length(255) == 8
        assert bit_length(256) == 9
    
    def test_digit_count(self):
        """Test digit count."""
        assert digit_count(0) == 1
        assert digit_count(9) == 1
        assert digit_count(10) == 2
        assert digit_count(99) == 2
        assert digit_count(100) == 3
        assert digit_count(10**99) == 100


class TestGmpy2Status:
    """Tests for gmpy2 availability reporting."""
    
    def test_have_gmpy2_is_bool(self):
        """HAVE_GMPY2 is a boolean."""
        assert isinstance(HAVE_GMPY2, bool)
    
    def test_functions_work_regardless(self):
        """All functions work whether gmpy2 is available or not."""
        # Just verify they don't crash
        assert isqrt(100) == 10
        assert gcd(12, 18) == 6
        assert is_prime(17)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

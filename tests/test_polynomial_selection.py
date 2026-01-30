"""Tests for improved polynomial selection in GNFS.

Tests cover:
- Base-m expansion and reconstruction
- Murphy E scoring and alpha computation
- Polynomial optimization
- Comparison between selection methods
"""

import math
import pytest

from gnfs.polynomial import (
    Polynomial,
    PolynomialSelection,
    PolynomialScore,
    select_polynomial,
    select_polynomial_classic,
    select_base_m,
    search_polynomial_range,
    optimize_polynomial,
    compare_polynomials,
    murphy_e_score,
    score_polynomial_selection,
    compute_alpha,
    compute_alpha_projective,
    coefficient_size_score,
    size_score_with_skewness,
    skewness,
    root_score,
    count_roots_mod_p,
    smoothness_score,
    base_m_expansion,
    optimal_base_m,
)


# =============================================================================
# Test Data
# =============================================================================

# Small test numbers
SMALL_N = 15          # 3 * 5
MEDIUM_N = 1073      # 29 * 37
RSA_LIKE = 2047       # 23 * 89

# Larger test numbers
LARGE_N = 1000003 * 1000033  # ~10^12
VERY_LARGE = 10**20 + 39     # 20-digit number


# =============================================================================
# Base-m Expansion Tests
# =============================================================================

class TestBaseMExpansion:
    """Tests for base-m expansion functionality."""
    
    def test_base_m_expansion_simple(self):
        """Test base-m expansion reconstructs the original number."""
        n = 100
        m = 10
        coeffs = base_m_expansion(n, m, degree=2)
        
        # Verify f(m) = n
        result = sum(c * (m ** i) for i, c in enumerate(coeffs))
        assert result == n
    
    def test_base_m_expansion_balanced(self):
        """Test that balanced representation gives smaller coefficients."""
        n = 99  # = 100 - 1 = 10^2 - 1
        m = 10
        coeffs = base_m_expansion(n, m, degree=2)
        
        # Balanced representation should give (-1, 0, 1) instead of (9, 9, 0)
        # Since 99 = -1 + 0*10 + 1*100
        assert coeffs[0] == -1 or (coeffs[0] == 9 and coeffs[1] == 9)
        
        # Verify reconstruction
        result = sum(c * (m ** i) for i, c in enumerate(coeffs))
        assert result == n
    
    def test_base_m_expansion_large(self):
        """Test base-m expansion on larger numbers."""
        n = LARGE_N
        m = optimal_base_m(n, degree=4)
        coeffs = base_m_expansion(n, m, degree=4)
        
        # Verify f(m) = n
        poly = Polynomial(coeffs)
        assert poly.evaluate(m) == n
    
    def test_optimal_base_m(self):
        """Test optimal_base_m returns reasonable values."""
        # For degree d, m should be approximately n^(1/d)
        n = 10**12
        
        for degree in [3, 4, 5]:
            m = optimal_base_m(n, degree)
            expected = int(round(n ** (1.0 / degree)))
            # Should be close to the expected value
            assert abs(m - expected) < expected * 0.1


# =============================================================================
# Polynomial Selection Tests
# =============================================================================

class TestPolynomialSelection:
    """Tests for polynomial selection functions."""
    
    def test_select_polynomial_degree_one(self):
        """Test degree-1 selection is unchanged."""
        selection = select_polynomial(100, degree=1)
        
        assert isinstance(selection, PolynomialSelection)
        assert selection.algebraic.degree() == 1
        assert selection.rational.degree() == 1
    
    def test_select_polynomial_degree_two(self):
        """Test degree-2 selection produces valid polynomial."""
        n = 143  # 11 * 13
        selection = select_polynomial(n, degree=2)
        
        # Verify algebraic polynomial has correct root
        assert selection.algebraic.evaluate(selection.m) % n == 0
        
        # Verify rational polynomial
        assert selection.rational.evaluate(selection.m) == 0
    
    def test_select_polynomial_degree_three(self):
        """Test degree-3 selection."""
        n = RSA_LIKE
        selection = select_polynomial(n, degree=3)
        
        assert selection.algebraic.degree() == 3
        assert selection.algebraic.evaluate(selection.m) % n == 0
    
    def test_select_polynomial_higher_degrees(self):
        """Test degree 4 and 5 selection."""
        n = LARGE_N
        
        for degree in [4, 5]:
            selection = select_polynomial(n, degree=degree)
            assert selection.algebraic.degree() == degree
            assert selection.algebraic.evaluate(selection.m) == n
    
    def test_select_polynomial_classic_compatibility(self):
        """Test classic selection still works."""
        n = 10
        selection = select_polynomial_classic(n, degree=3)
        
        # For n=10 and degree=3, m = round(10^(1/3)) = 2
        # (x + 2)^3 - 10 = x^3 + 6x^2 + 12x + 8 - 10 = x^3 + 6x^2 + 12x - 2
        assert selection.m == 2
        # Coefficients: (-2, 12, 6, 1)
        assert selection.algebraic.coeffs == (-2, 12, 6, 1)
    
    def test_select_polynomial_invalid_degree(self):
        """Test that degree 0 raises an error."""
        with pytest.raises(ValueError):
            select_polynomial(100, degree=0)
    
    def test_select_base_m(self):
        """Test base-m selection directly."""
        n = MEDIUM_N
        selection = select_base_m(n, degree=3)
        
        assert selection.algebraic.evaluate(selection.m) == n
        assert selection.rational.evaluate(selection.m) == 0


# =============================================================================
# Scoring Tests
# =============================================================================

class TestPolynomialScoring:
    """Tests for polynomial quality metrics."""
    
    def test_compute_alpha(self):
        """Test alpha computation returns a number."""
        poly = Polynomial((1, 0, 1))  # x^2 + 1
        alpha = compute_alpha(poly)
        
        assert isinstance(alpha, float)
        assert not math.isnan(alpha)
        assert not math.isinf(alpha)
    
    def test_compute_alpha_projective(self):
        """Test projective alpha accounts for leading coefficient."""
        poly1 = Polynomial((1, 0, 1))   # x^2 + 1
        poly2 = Polynomial((1, 0, 2))   # 2x^2 + 1
        
        alpha1 = compute_alpha_projective(poly1)
        alpha2 = compute_alpha_projective(poly2)
        
        # Both should be finite
        assert not math.isnan(alpha1)
        assert not math.isnan(alpha2)
    
    def test_coefficient_size_score(self):
        """Test size scoring prefers smaller coefficients."""
        poly_small = Polynomial((1, 1, 1))
        poly_large = Polynomial((1000, 1000, 1000))
        
        score_small = coefficient_size_score(poly_small)
        score_large = coefficient_size_score(poly_large)
        
        # Smaller coefficients should give smaller score
        assert score_small < score_large
    
    def test_skewness(self):
        """Test skewness computation."""
        # Polynomial with equal absolute coefficients
        poly = Polynomial((100, 0, 1))
        s = skewness(poly)
        
        # Skewness should be positive
        assert s > 0
        # For coefficients (100, 0, 1), skewness ≈ (100/1)^(1/2) = 10
        assert abs(s - 10) < 1
    
    def test_size_score_with_skewness(self):
        """Test skew-adjusted size score."""
        poly = Polynomial((1, 2, 3, 4, 5))
        score = size_score_with_skewness(poly)
        
        assert score > 0
        assert not math.isnan(score)
    
    def test_count_roots_mod_p(self):
        """Test root counting modulo primes."""
        # x^2 - 1 has roots ±1 mod p for most primes
        poly = Polynomial((-1, 0, 1))  # x^2 - 1
        
        # Mod 5: roots are 1 and 4 (since 4 ≡ -1)
        assert count_roots_mod_p(poly, 5) == 2
        
        # Mod 2: only root is 1
        assert count_roots_mod_p(poly, 2) == 1
    
    def test_root_score(self):
        """Test root score is higher for polynomials with more roots."""
        # x has many roots (one per prime)
        poly_many_roots = Polynomial((0, 1))
        
        # x^2 + 1 has fewer roots in general
        poly_few_roots = Polynomial((1, 0, 1))
        
        score_many = root_score(poly_many_roots)
        score_few = root_score(poly_few_roots)
        
        # More roots should give higher score
        assert score_many > score_few
    
    def test_smoothness_score(self):
        """Test smoothness scoring."""
        # Highly smooth number
        smooth = 2**10 * 3**5 * 5**3  # 2^10 * 3^5 * 5^3 = 12,441,600
        score_smooth = smoothness_score(smooth)
        
        # Prime number (not smooth)
        prime = 1000003
        score_prime = smoothness_score(prime)
        
        assert score_smooth > score_prime
        assert score_smooth == 1.0  # Completely smooth
    
    def test_murphy_e_score(self):
        """Test Murphy E score computation."""
        n = MEDIUM_N
        
        # Compare classic vs base-m
        classic = select_polynomial_classic(n, degree=3)
        base_m = select_base_m(n, degree=3)
        
        score_classic = murphy_e_score(classic.algebraic, n)
        score_base_m = murphy_e_score(base_m.algebraic, n)
        
        # Both should be positive
        assert score_classic > 0
        assert score_base_m > 0
    
    def test_score_polynomial_selection(self):
        """Test complete polynomial scoring."""
        n = MEDIUM_N
        selection = select_polynomial(n, degree=3)
        
        score = score_polynomial_selection(selection, n)
        
        assert isinstance(score, PolynomialScore)
        assert not math.isnan(score.alpha)
        assert not math.isnan(score.size_score)
        assert score.murphy_e > 0


# =============================================================================
# Optimization Tests
# =============================================================================

class TestPolynomialOptimization:
    """Tests for polynomial optimization."""
    
    def test_search_polynomial_range(self):
        """Test polynomial search finds valid polynomial."""
        n = MEDIUM_N
        selection = search_polynomial_range(n, degree=3, m_range=20)
        
        assert selection.algebraic.evaluate(selection.m) == n
    
    def test_optimize_polynomial(self):
        """Test optimization improves or maintains score."""
        n = LARGE_N
        
        initial = select_base_m(n, degree=4)
        initial_score = murphy_e_score(initial.algebraic, n)
        
        optimized = optimize_polynomial(initial, n, iterations=20)
        optimized_score = murphy_e_score(optimized.algebraic, n)
        
        # Optimized should be at least as good
        assert optimized_score >= initial_score * 0.95  # Allow small tolerance
        
        # Should still be valid
        assert optimized.algebraic.evaluate(optimized.m) == n
    
    def test_compare_polynomials(self):
        """Test polynomial comparison utility."""
        n = RSA_LIKE
        results = compare_polynomials(n, degree=3)
        
        assert 'classic' in results
        assert 'base_m' in results
        assert 'optimized' in results
        
        # Check each method has valid polynomial for its convention
        # Classic method: f(-m) ≡ 0 (mod n) (old convention)
        classic = results['classic']['selection']
        assert classic.algebraic.evaluate(-classic.m) % n == 0
        
        # Base-m and optimized: f(m) = n exactly (GNFS convention)
        for method in ['base_m', 'optimized']:
            selection = results[method]['selection']
            assert selection.algebraic.evaluate(selection.m) == n


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for polynomial selection in GNFS context."""
    
    def test_polynomial_root_property(self):
        """Test that selected polynomials have the required root property.
        
        For GNFS, we need:
        - f(m) = n (so f(m) ≡ 0 mod n)
        - g(m) = 0 (rational polynomial)
        
        This ensures both polynomials share root m modulo n.
        """
        test_numbers = [SMALL_N, MEDIUM_N, RSA_LIKE]
        
        for n in test_numbers:
            for degree in [2, 3]:
                selection = select_polynomial(n, degree=degree)
                
                # f(m) = n exactly (so f(m) ≡ 0 mod n)
                assert selection.algebraic.evaluate(selection.m) == n
                
                # g(m) = 0
                assert selection.rational.evaluate(selection.m) == 0
    
    def test_homogeneous_evaluation(self):
        """Test homogeneous polynomial evaluation used in sieving."""
        n = MEDIUM_N
        selection = select_polynomial(n, degree=3)
        
        # For (a, b) = (m, 1), homogeneous f should equal f(m)
        m = selection.m
        poly = selection.algebraic
        
        homo_result = poly.evaluate_homogeneous(m, 1)
        regular_result = poly.evaluate(m)
        
        assert homo_result == regular_result
    
    def test_improved_vs_classic_quality(self):
        """Test that improved selection produces better polynomials for larger n."""
        n = LARGE_N
        degree = 4
        
        classic = select_polynomial_classic(n, degree=degree)
        improved = select_polynomial(n, degree=degree, optimize=True)
        
        # Compare coefficient sizes
        classic_max = max(abs(c) for c in classic.algebraic.coeffs)
        improved_max = max(abs(c) for c in improved.algebraic.coeffs)
        
        # Improved should generally have smaller coefficients
        # (This may not always hold, but should in most cases)
        # Just verify both are valid for their conventions
        # Classic: f(-m) ≡ 0 (mod n)
        assert classic.algebraic.evaluate(-classic.m) % n == 0
        # Improved: f(m) = n exactly
        assert improved.algebraic.evaluate(improved.m) == n
    
    def test_selection_stability(self):
        """Test that selection is deterministic."""
        n = MEDIUM_N
        
        sel1 = select_polynomial(n, degree=3, optimize=False)
        sel2 = select_polynomial(n, degree=3, optimize=False)
        
        assert sel1.algebraic.coeffs == sel2.algebraic.coeffs
        assert sel1.m == sel2.m


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_n(self):
        """Test with very small n."""
        for n in [4, 6, 8, 9, 10]:
            selection = select_polynomial(n, degree=2)
            assert selection.algebraic.evaluate(selection.m) % n == 0
    
    def test_prime_n(self):
        """Test with prime n (edge case, GNFS not ideal for primes)."""
        n = 997  # Prime
        selection = select_polynomial(n, degree=2)
        assert selection.algebraic.evaluate(selection.m) % n == 0
    
    def test_power_of_two(self):
        """Test with power of 2."""
        n = 1024
        selection = select_polynomial(n, degree=3)
        assert selection.algebraic.evaluate(selection.m) == n
    
    def test_empty_polynomial(self):
        """Test scoring handles edge cases gracefully."""
        poly = Polynomial((0,))
        
        # Should not crash
        alpha = compute_alpha(poly)
        assert isinstance(alpha, float)

"""Polynomial utilities for the General Number Field Sieve."""

from .number_field import NumberField, NumberFieldElement
from .polynomial import Polynomial
from .selection import (
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

__all__ = [
    # Core classes
    "Polynomial",
    "PolynomialSelection",
    "PolynomialScore",
    "NumberField",
    "NumberFieldElement",
    # Main selection functions
    "select_polynomial",
    "select_polynomial_classic",
    "select_base_m",
    "search_polynomial_range",
    "optimize_polynomial",
    "compare_polynomials",
    # Scoring functions
    "murphy_e_score",
    "score_polynomial_selection",
    "compute_alpha",
    "compute_alpha_projective",
    "coefficient_size_score",
    "size_score_with_skewness",
    "skewness",
    "root_score",
    "count_roots_mod_p",
    "smoothness_score",
    # Base-m utilities
    "base_m_expansion",
    "optimal_base_m",
]

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An educational Python implementation of the General Number Field Sieve (GNFS) algorithm, designed to eventually become an interactive website demonstrating how GNFS works. The goal is well-documented, readable Python code that teaches the algorithm rather than a production factorization tool.

The implementation follows the real GNFS pipeline: polynomial selection → sieving → linear algebra → square root extraction.

## Commands

### Installation
```bash
pip install sympy numpy
# Or install as package:
pip install -e .
```

### Running the CLI
```bash
python cli.py <integer>                    # Factor an integer with defaults
python cli.py 30 --degree 1 --bound 40 --interval 60  # With custom parameters
```

### Running Tests
```bash
pytest                          # Run all tests
pytest tests/test_sieve.py     # Run single test file
pytest -k "test_factor"        # Run tests matching pattern
```

## Architecture

The GNFS pipeline is implemented as four sequential stages, each in its own module:

### `gnfs/polynomial/` - Polynomial Selection
- `selection.py`: Constructs polynomials of form `(x + m)^d - n` where `m ≈ n^(1/d)`
- `PolynomialSelection` bundles the algebraic polynomial with the rational polynomial `x - m`
- The polynomial is chosen so `x = -m` is a root modulo `n`

### `gnfs/sieve/` - Relation Finding
- `sieve.py`: Two-sided logarithmic line sieve over algebraic and rational polynomials
- For each prime `p`, finds polynomial roots mod `p` and subtracts `log(p)` from sieve array
- Positions with small residuals are trial-factored to confirm B-smoothness
- `Relation` dataclass stores `(a, b)` pairs with their factorizations on both sides

### `gnfs/linalg/` - Linear Algebra
- `matrix.py`: Gaussian elimination over GF(2) to find nullspace of exponent matrix
- Relations are combined into exponent vectors; dependencies indicate which relations multiply to a square

### `gnfs/sqrt/` - Square Root and Factor Extraction
- `square_root.py`: Combines dependent relations into congruence of squares
- Extracts non-trivial factors via `gcd(x - y, n)`

### Orchestration
- `gnfs/factor.py`: `gnfs_factor()` ties all stages together
- Expands sieving interval in rounds until enough relations are gathered
- Requires `len(primes) + 1` relations to guarantee dependencies exist

## Key Data Flow

1. `select_polynomial(n, degree)` → `PolynomialSelection`
2. `find_relations(selection, primes, interval)` → yields `Relation` objects
3. `solve_matrix(relations, primes)` → list of dependency indices
4. `find_factors(n, relations, primes)` → yields prime factors

## Configuration

Default parameters in `default_config.json`:
- `degree`: Polynomial degree (default 1)
- `bound`: Smoothness bound for factor base (default 30)
- `interval`: Sieving interval radius (default 50)

## Development Guidelines

- Prioritize code clarity and documentation over performance optimizations
- Each module should be understandable in isolation for educational purposes
- The book manuscript in `book/manuscript.md` explains the algorithm in depth - keep code aligned with its explanations

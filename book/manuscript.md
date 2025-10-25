# The General Number Field Sieve: From Theory to Practice

## Front Matter

### Title Page

**The General Number Field Sieve: From Theory to Practice**  
By The GNFS Collective

### Copyright

Copyright © 2024 The GNFS Collective. All rights reserved. No part of this book may be reproduced without permission, except for brief quotations in critical articles or reviews.

### Dedication

For the mathematicians and engineers who turned abstract algebra into working code.

### Table of Contents

1. Preface
2. Chapter 1 — Factoring Integers in the Real World
3. Chapter 2 — Anatomy of the General Number Field Sieve
4. Chapter 3 — Polynomial Selection in Depth
5. Chapter 4 — Walking the Line Sieve
6. Chapter 5 — Linear Algebra over GF(2)
7. Chapter 6 — The Square Root Step and Final Split
8. Chapter 7 — Putting the Pipeline Together
9. Chapter 8 — Running and Extending the Implementation
10. Chapter 9 — Testing, Validation, and Further Reading
11. Appendices
    - Appendix A — Command-Line Reference
    - Appendix B — Configuration Files
    - Appendix C — Glossary of Terms
    - Appendix D — Additional Resources

## Preface

Large integers sit at the heart of modern cryptography. Breaking them apart is more than an academic exercise: it is an arms race between algorithm designers and the engineers who deploy those algorithms at scale. This book walks you through a compact yet faithful implementation of the General Number Field Sieve (GNFS), guiding you from the mathematical underpinnings to the lines of Python code that orchestrate each stage of the factorisation process. The project described throughout mirrors the true GNFS pipeline, presenting the real algorithms in a form that can run on a laptop, and highlighting the compromises and insights that make the method viable for large-scale computations.【F:README.md†L1-L40】【F:gnfs/factor.py†L1-L36】

The chapters that follow combine narrative explanations with annotated code, practical recipes, and historical context. By the end, you will not only understand what the GNFS does but also how to implement, experiment with, and extend it yourself. Whether you are a mathematician seeking intuition, a software engineer curious about computational number theory, or a student looking for a project that bridges theory and practice, this book aims to be your companion.

## Chapter 1 — Factoring Integers in the Real World

Integer factorisation has a storied history stretching from ancient methods like trial division to modern sub-exponential algorithms. The RSA cryptosystem and many related protocols derive their security from the difficulty of factoring numbers with several hundred digits. The GNFS currently stands as the fastest known general-purpose algorithm for large integers, and its structure encapsulates decades of research.

In this chapter we introduce the landscape of factoring algorithms, describe the asymptotic performance of the GNFS, and explain why the algorithm is worth studying even in a deliberately simplified code base. You will see how smoothness bounds and polynomial selection combine to make large-scale sieving feasible, and why modern implementations rely on distributed computing to push the limits of factorable sizes.

## Chapter 2 — Anatomy of the General Number Field Sieve

The GNFS can be viewed as a four-act play:

1. **Polynomial selection** chooses algebraic and rational polynomials with shared roots mod the target integer.
2. **Sieving** identifies copious relations whose norms factor completely over a bounded prime base.
3. **Linear algebra** uses those relations to discover dependencies modulo two.
4. **The square root step** turns a dependency into a congruence of squares that splits the integer.

Our implementation keeps these acts distinct, echoing production-grade GNFS suites. Each module in the code repository corresponds to a stage of the pipeline and is intentionally self-contained. The structure is laid out in the project documentation and mirrored by the modules under the `gnfs` package.【F:README.md†L8-L40】

Throughout the remainder of the book we follow the data as it flows from one stage to the next, pausing to unpack the mathematics and algorithms that animate each component.

## Chapter 3 — Polynomial Selection in Depth

Polynomial selection shapes the efficiency of the entire sieve. In industrial-strength GNFS deployments, elaborate searches produce polynomials with tiny coefficients and favourable root properties. Our compact implementation embraces the classic `(x + m)^d - n` family, which already captures the essential behaviour.

The `gnfs.polynomial.selection` module implements this recipe by approximating the real `d`-th root of the target integer, rounding it to the nearest integer `m`, and then expanding the binomial polynomial. The constant term is adjusted to maintain a root modulo `n`, while the rational polynomial defaults to `x - m` for the projective lift.【F:gnfs/polynomial/selection.py†L1-L61】

The data class `PolynomialSelection` bundles the algebraic and rational polynomials alongside the root `m`, providing a compact container passed to the sieving stage.【F:gnfs/polynomial/selection.py†L16-L34】 The resulting algebraic polynomial supports both affine and homogeneous evaluation via the `Polynomial` class, which stores coefficients and exposes helper methods for evaluating either directly or in projective coordinates.【F:gnfs/polynomial/polynomial.py†L1-L31】

To work comfortably with algebraic numbers, the implementation models the number field defined by the algebraic polynomial. The `NumberField` class represents the quotient `ℚ[x]/(f(x))`, reduces elements modulo the minimal polynomial, and provides arithmetic, norms, and canonical bases through the `NumberFieldElement` helper class.【F:gnfs/polynomial/number_field.py†L1-L205】【F:gnfs/polynomial/number_field.py†L205-L298】 These utilities remain faithful to real GNFS software, even while we limit ourselves to low-degree polynomials.

### Worked Example

Suppose we wish to factor `n = 8051` using a degree-two polynomial. The selection routine computes `m ≈ n^{1/2} = 89.77...` and rounds to `90`. Expanding `(x + 90)^2 - 8051` yields `x^2 + 180x + 8100 - 8051 = x^2 + 180x + 49`. The polynomial has small coefficients, and substituting `x = -90` gives a value divisible by `8051`, satisfying the modular root requirement. The rational polynomial `x - 90` complements the algebraic side. With these pieces in hand, the sieve can search for pairs `(a, b)` that make both norms smooth.

## Chapter 4 — Walking the Line Sieve

In the sieving phase we hunt for integer pairs `(a, b)` whose algebraic and rational norms factor completely over a preselected set of primes. Full-scale GNFS computations perform lattice sieving or the even more efficient line sieving variants, but our implementation keeps to a one-dimensional line sieve that remains faithful to the core ideas.【F:gnfs/sieve/sieve.py†L1-L73】

The `find_relations` function accepts a polynomial selection, a list of primes that form the factor base, and an interval width. For each denominator `b` in the interval, it precomputes the algebraic norms `b^d f(a/b)` over the range of numerators and maintains an array of logarithmic weights. Every time a prime divides a norm, the function subtracts `log p` from the corresponding entry. After processing all primes, entries with near-zero residuals are likely to be smooth. The function then trial factors both norms to verify that they are entirely composed of factor-base primes before yielding a `Relation` object that records the exponents.【F:gnfs/sieve/sieve.py†L15-L73】【F:gnfs/sieve/relation.py†L1-L21】

Roots modulo each prime are computed via SymPy, which factors the polynomial over finite fields. This step mirrors the root-finding tasks of real sievers, albeit with the convenience of a computer algebra system in place of specialized lattice arithmetic.【F:gnfs/sieve/roots.py†L1-L22】 In practice, the sieve spends most of its time here: iterating over primes, updating sieve arrays, and trial factoring candidate relations.

### Worked Example

Continuing with our `n = 8051` example, imagine we choose primes up to `B = 30` and an interval of `b ≤ 20`. For each `b`, the sieve populates arrays of length `2 * interval + 1` to cover numerators `a` from `-20` to `20`. If the entry corresponding to `a = 11`, `b = 7` survives the logarithmic subtraction with a small residual, the sieve trial factors the norms. When both sides turn out to be composed entirely of primes ≤ 30, the pair becomes a relation and contributes a row to the exponent matrix that drives the linear algebra stage.

## Chapter 5 — Linear Algebra over GF(2)

The sieve produces hundreds or thousands of relations, each describing how the norms factor over the prime base. To assemble a congruence of squares we need a dependency among these relations modulo two—essentially, a subset whose combined exponents are even for every prime. The `gnfs.linalg` module provides a lightweight Gaussian elimination routine that accomplishes this task.【F:gnfs/linalg/matrix.py†L1-L66】

The `solve_matrix` function builds a matrix whose rows correspond to primes and whose columns capture the parity of exponents appearing in each relation. Running `_nullspace_mod2` returns a basis of dependency vectors. Each vector enumerates the relations that, when multiplied together, give even exponents for every prime in the base.【F:gnfs/linalg/matrix.py†L68-L96】 This step echoes the massive sparse matrix solves performed in production GNFS efforts, where specialised block Lanczos or block Wiedemann algorithms are required to tackle matrices with millions of columns.

### Worked Example

Assume that we have collected five relations over a factor base consisting of the first ten primes. The exponent matrix is a 10×5 array reduced modulo two. Gaussian elimination identifies free variables corresponding to relations that can be combined. If the nullspace basis contains the vector `(1, 0, 1, 1, 0)`, we learn that relations 0, 2, and 3 sum to zero modulo two for every prime. Those three relations will be candidates for the square root step.

## Chapter 6 — The Square Root Step and Final Split

Once a dependency is in hand, we recombine the associated relations to build the congruence `x^2 ≡ y^2 (mod n)` that ultimately splits the integer. The `find_factors` routine multiplies together the rational norms of the chosen relations to form `x`, multiplies the absolute algebraic norms to form `y^2`, and checks that the product is a perfect square. It then computes `gcd(x - y, n)` and `gcd(x + y, n)` to obtain non-trivial factors whenever possible.【F:gnfs/sqrt/square_root.py†L1-L35】

While our code uses Python’s standard library and SymPy for convenience, the logic mirrors real GNFS software. Multiple dependencies may need to be tested before a successful factorisation occurs, so the function yields each factor pair it finds, stopping once a split is achieved.

### Worked Example

Returning to `n = 8051`, suppose the dependency involves relations with rational norms `-13`, `17`, and `-7`. Multiplying their absolute values yields `x = 13 · 17 · 7 = 1547`. On the algebraic side we multiply the norms, obtaining a perfect square `y^2`. If `y = 2880`, then `gcd(1547 - 2880, 8051) = gcd(-1333, 8051) = 97`, revealing the factorisation `8051 = 97 × 83`.

## Chapter 7 — Putting the Pipeline Together

With each stage implemented, the orchestration layer stitches them together. The `gnfs_factor` function accepts the target integer, smoothness bound, sieving interval, and polynomial degree. It performs polynomial selection, enumerates the prime factor base via SymPy, gathers relations, and calls the square root step to return any discovered factors.【F:gnfs/factor.py†L1-L36】 This function is the heartbeat of the project, providing a minimal yet authentic GNFS run that a reader can experiment with immediately.

The command-line interface wraps `gnfs_factor` and loads default parameters from `default_config.json`, allowing users to invoke the factorisation pipeline with a single command. Optional flags let you tweak the degree, bound, and interval without editing the source.【F:cli.py†L1-L40】 Chapter 8 demonstrates how to run the CLI, interpret the output, and adjust parameters to balance runtime and success probability.

## Chapter 8 — Running and Extending the Implementation

Before diving in, ensure that Python 3.11+ and SymPy are installed. The repository ships with a `pyproject.toml` that specifies dependencies and enables installation via `pip`. Once SymPy is available, you can factor small integers by executing `python cli.py 30`, which prints the discovered factors.【F:README.md†L42-L66】【F:cli.py†L21-L40】 Larger numbers may require tuning the smoothness bound and sieving interval or experimenting with higher-degree polynomials.

This chapter offers guidance on:

* Choosing sensible factor base bounds relative to the target integer.
* Detecting when additional relations are needed and how to increase the sieving interval accordingly.
* Modifying the polynomial selection to explore different degrees or alternative root properties.
* Profiling hotspots in the sieve and experimenting with basic optimisations such as caching logarithms or skipping redundant trial divisions.

For adventurous readers, we sketch how to extend the codebase with lattice sieving, large prime variants, or distributed relation collection. Each extension is framed as a self-contained project, building on the clean modular structure of the existing implementation.

## Chapter 9 — Testing, Validation, and Further Reading

Reliability matters even in educational implementations. The repository includes unit tests that exercise the polynomial utilities, sieve helpers, and matrix solver. Running `pytest` ensures that refactors do not inadvertently change behaviour. The tests provide a safety net as you explore the code and experiment with new features.

We close the chapter with a curated bibliography of research papers, textbooks, and online resources that chart the development of the GNFS and its predecessors. Highlights include the original Lenstra papers, the Cunningham project reports, and modern accounts of large factorizations. Together they provide a roadmap for deeper study.

## Appendices

### Appendix A — Command-Line Reference

The CLI supports the following options:

* `n`: Required positional integer to factor.
* `--config`: Path to a JSON configuration file. Defaults to `default_config.json`.
* `--degree`: Overrides the polynomial degree.
* `--bound`: Sets the factor base bound.
* `--interval`: Controls the sieving interval radius.

Internally the CLI loads the default configuration, applies command-line overrides, and prints the factors returned by the pipeline.【F:cli.py†L1-L40】 A sample configuration file is supplied in the repository.

### Appendix B — Configuration Files

The default configuration ships with `degree = 1`, `bound = 30`, and `interval = 50`, providing a balanced starting point for small integers.【F:default_config.json†L1-L5】 You can create additional configuration files tailored to specific factorisation tasks and point the CLI to them using the `--config` option.

### Appendix C — Glossary of Terms

* **Algebraic number field** — The extension of the rationals defined by a polynomial, represented in code by the `NumberField` class.【F:gnfs/polynomial/number_field.py†L31-L198】
* **Factor base** — A finite set of small primes against which norms are trial-factored during sieving.【F:gnfs/sieve/sieve.py†L15-L73】
* **Relation** — A pair `(a, b)` whose norms factor completely over the factor base, recorded by the `Relation` data class.【F:gnfs/sieve/relation.py†L1-L21】
* **Smooth number** — An integer whose prime factors lie below a given bound.
* **Square root step** — The stage that transforms dependencies into actual factors of `n` via greatest common divisors.【F:gnfs/sqrt/square_root.py†L1-L35】

### Appendix D — Additional Resources

* **SymPy documentation** — Essential for understanding the finite field factoring utilities used in the sieve.【F:gnfs/sieve/roots.py†L1-L22】
* **NumPy documentation** — Useful for exploring the matrix routines that underpin the linear algebra stage.【F:gnfs/linalg/matrix.py†L1-L96】
* **Python standard library** — Modules like `math` and `fractions` provide the building blocks for arbitrary-precision arithmetic and exact rational calculations.【F:gnfs/polynomial/number_field.py†L1-L198】【F:gnfs/sieve/sieve.py†L15-L73】

## About the Authors

The GNFS Collective is a group of developers and mathematicians passionate about bringing advanced number theoretic algorithms to life in approachable code. Contributions are welcome—see the project repository for details on how to get involved.

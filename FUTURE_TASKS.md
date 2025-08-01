# Future Tasks for a Full GNFS Implementation

The current code base offers only a minimal demonstration. The tasks below break down the remaining work into sessions that can be completed and merged one after another.

## 1. Prepare the Project
- ~~Create a `pyproject.toml` and package the code under the `gnfs` namespace.~~
- ~~Set up continuous integration running `pytest`.~~

## 2. Restructure Modules
- ~~Split `gnfs` into subpackages: `polynomial`, `sieve`, `linalg`, and `sqrt`.~~
- ~~Update imports and tests to match the new layout.~~

## 3. Improve the CLI
- ~~Add command line options for polynomial degree, factor base size, and sieve range.~~
- ~~Provide a default configuration file.~~

## 4. Implement Polynomial Selection
- Replace the temporary `x**degree - n` polynomial generation with a
  search for polynomials that minimize coefficient size.
- Implement heuristics to choose the optimal degree for the input
  integer.
- Add root optimization to favor smooth values and improve sieving
  efficiency.
- Create a `PolynomialFinder` class that encapsulates the search logic.
- Start with a stub that returns the current placeholder polynomial so
  existing tests continue to pass.
- Extend the finder to enumerate candidate parameter sets within user
  defined bounds and score them with Murphy's `E` metric.
- Update `select_polynomial` to call the finder once it can produce
  better polynomials.
- Add unit tests exercising the finder interface and scoring functions.

## 5. Expand Sieving
- Introduce rational and algebraic sieving routines.
- Support multiprocessing for larger intervals.

## 6. Handle Relations
- Record both rational and algebraic factors in a new `Relation` object.
- Store relations on disk and filter duplicates.

## 7. Large-Prime Variant
- Allow relations to contain one large prime and recombine partial relations.

## 8. Scalable Linear Algebra
- Replace the current elimination with a block Lanczos solver on sparse matrices.
- Add checkpointing so long jobs can resume.

## 9. Square Root Phase
- Compute square roots in the number field as well as over the integers.
- Assemble the final congruence of squares and extract factors.

## 10. Optimisation and Cleanup
- Use `gmpy2` for big integers and profile performance.
- Expand documentation and examples in the README.

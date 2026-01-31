# GNFS Production Roadmap

**Goal:** Transform this educational implementation into a production-quality GNFS that can factor cryptographic-size integers (100+ digits) entirely in Python.

## Current State

✅ Working 4-stage pipeline  
✅ Correct mathematical foundations  
✅ Educational documentation  
⚠️ Only works on small numbers (~20 digits max)  
⚠️ No optimizations  
⚠️ Single-threaded  

## Phase 1: Core Algorithm Improvements

### 1.1 Polynomial Selection (Priority: HIGH) ✅ PARTIALLY COMPLETE
- [x] Add polynomial rating/scoring (alpha, Murphy E)
- [x] Base-m expansion for balanced coefficients
- [x] Size optimization for coefficients (skewness-adjusted)
- [x] Support for degree 4, 5, 6 polynomials
- [x] Root counting and smoothness analysis
- [ ] Implement full Kleinjung's polynomial selection algorithm
- [ ] Advanced root optimization techniques

**Implemented (2025):**
- Murphy E scoring combining alpha, size, root properties, and smoothness
- Base-m expansion replacing naive (x+m)^d-n construction
- Coefficient optimization with search and local refinement
- Mathematical correctness fix: f(m) = n (proper GNFS root property)
- Comprehensive test suite (32 new tests)

**Why:** Good polynomials can speed up sieving by 10-100x. The improvements provide better polynomial quality for larger numbers.

### 1.2 Lattice Sieving (Priority: HIGH) ✅ PARTIALLY COMPLETE
- [x] Replace line sieve with lattice sieve
- [x] Implement special-q sieving
- [x] Smart special-q selection (primes with polynomial roots)
- [x] Lattice basis computation and Lagrange reduction
- [ ] Factor base optimization
- [ ] Bucket sieving for cache efficiency
- [ ] Large prime variations (2LP, 3LP)

**Implemented (2026):**
- Special-q lattice sieving: for each special prime q, sieve the sublattice L_q = {(a,b) : a ≡ rb (mod q)}
- Two sieve implementations: basic trial division and optimized logarithmic sieving
- Smart special-q selection that filters for primes with polynomial roots
- Lattice basis computation with Lagrange reduction for shorter vectors
- Hybrid sieve that automatically chooses between line and lattice sieve
- Comprehensive test suite

**Benchmark results (n=2021, 25 primes):**
- Line sieve: 1 relation
- Lattice sieve: 44 relations
- The lattice sieve finds many more relations by exploring multiple sublattices

**Why:** Lattice sieving is exponentially faster than line sieving for large numbers.

### 1.3 Linear Algebra (Priority: HIGH)
- [ ] Implement Block Lanczos algorithm
- [ ] Or Block Wiedemann (better for distributed)
- [ ] Sparse matrix representation
- [ ] Structured Gaussian elimination as preprocessing

**Why:** Current O(n³) Gaussian elimination won't scale. Block Lanczos is O(n²) with small constants.

### 1.4 Square Root (Priority: MEDIUM)
- [ ] Montgomery's square root algorithm
- [ ] Handle algebraic square roots properly
- [ ] Couveignes' algorithm for number field elements

**Why:** Current implementation is correct but inefficient for large algebraic integers.

## Phase 2: Performance & Scale

### 2.1 Large Number Arithmetic ✅ COMPLETE
- [x] Integrate `gmpy2` for GMP bindings (optional dependency)
- [x] Fallback to pure Python for portability
- [x] Efficient modular arithmetic (powmod, mod_inverse)
- [x] Number-theoretic utilities (Jacobi, primality, factorization)

**Implemented (2026):**
- Unified `gnfs.arith` module with automatic gmpy2 detection
- All functions work with or without gmpy2 (10-100x speedup with gmpy2)
- Functions: mpz, isqrt, iroot, gcd, lcm, mod_inverse, jacobi, is_prime, next_prime, prev_prime, is_power, factor_trial, powmod, primes_up_to
- Deterministic Miller-Rabin for small numbers, probabilistic for large
- 53 tests covering all functionality

### 2.2 Parallelization
- [ ] Multi-threaded sieving (embarrassingly parallel)
- [ ] Parallel polynomial selection
- [ ] Distributed linear algebra support
- [ ] `multiprocessing` / `concurrent.futures` integration

### 2.3 Memory Optimization
- [ ] Streaming relation collection
- [ ] Memory-mapped files for large matrices
- [ ] Compressed relation storage
- [ ] Checkpointing for long runs

## Phase 3: Production Features

### 3.1 Robustness
- [ ] Automatic parameter selection based on input size
- [ ] Progress reporting and ETA
- [ ] Graceful handling of edge cases
- [ ] Comprehensive error messages

### 3.2 Usability
- [ ] `pip install gnfs` distribution
- [ ] CLI with progress bars
- [ ] Python API with sensible defaults
- [ ] Jupyter notebook integration

### 3.3 Testing & Validation
- [ ] Test against known factorizations
- [ ] Benchmark suite
- [ ] Comparison with CADO-NFS / msieve
- [ ] CI/CD pipeline

## Milestones

| Milestone | Target | Description |
|-----------|--------|-------------|
| **v0.2** | 40-digit numbers | Improved polynomial selection |
| **v0.3** | 60-digit numbers | Lattice sieving |
| **v0.4** | 80-digit numbers | Block Lanczos |
| **v0.5** | 100-digit numbers | Multi-threading |
| **v1.0** | 120+ digits | Production ready |

## Technical Decisions

### Pure Python vs C Extensions
**Decision:** Pure Python with optional `gmpy2` for speed.

**Rationale:** 
- Maximizes portability and readability
- `gmpy2` provides 10-100x speedup for big integer ops
- NumPy/SciPy for matrix operations where beneficial
- Numba JIT compilation for hot loops (optional)

### Target Performance
For a 100-digit semiprime on modern hardware (single machine):
- Polynomial selection: < 1 hour
- Sieving: < 1 week  
- Linear algebra: < 1 day
- Square root: < 1 hour

This is ~10x slower than CADO-NFS (written in C), which is acceptable for a Python implementation.

## Resources

### Papers
- Kleinjung, T. (2006). "On polynomial selection for the general number field sieve"
- Franke & Kleinjung (2005). "Continued fractions and lattice sieving"
- Montgomery (1994). "Square roots of products of algebraic numbers"
- Coppersmith (1993). "Modifications to the Number Field Sieve"

### Reference Implementations
- [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) - Gold standard, C
- [msieve](https://github.com/radii/msieve) - Highly optimized, C
- [yafu](https://github.com/bbuhrow/yafu) - Feature-rich, C
- [sympy.ntheory](https://docs.sympy.org/latest/modules/ntheory.html) - Python, educational

## Contributing

Each phase can be worked on independently. Good first issues:
1. Implement Murphy E scoring for polynomials
2. Add `gmpy2` support with fallback
3. Parallelize relation collection
4. Add progress reporting to CLI

---

*This roadmap will evolve as we learn more about the performance characteristics of each component.*

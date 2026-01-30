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

### 1.1 Polynomial Selection (Priority: HIGH)
- [ ] Implement Kleinjung's polynomial selection algorithm
- [ ] Add polynomial rating/scoring (alpha, Murphy E)
- [ ] Root optimization for algebraic polynomial
- [ ] Size optimization for coefficients
- [ ] Support for degree 4, 5, 6 polynomials

**Why:** Good polynomials can speed up sieving by 10-100x. Current naive selection is a major bottleneck.

### 1.2 Lattice Sieving (Priority: HIGH)
- [ ] Replace line sieve with lattice sieve
- [ ] Implement special-q sieving
- [ ] Factor base optimization
- [ ] Bucket sieving for cache efficiency
- [ ] Large prime variations (2LP, 3LP)

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

### 2.1 Large Number Arithmetic
- [ ] Integrate `gmpy2` for GMP bindings (optional dependency)
- [ ] Fallback to pure Python for portability
- [ ] Efficient modular arithmetic
- [ ] Number-theoretic transforms for multiplication

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

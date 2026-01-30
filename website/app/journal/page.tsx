export default function JournalPage() {
  const entries = [
    {
      date: '2026-01-30',
      title: 'Phase 1.3: Block Lanczos Algorithm',
      status: 'complete',
      content: `Implemented Block Lanczos algorithm for O(n²) nullspace computation over GF(2).

**What was implemented:**
- **SparseMatrixGF2** — Memory-efficient sparse matrix storing only positions of 1s, with fast GF(2) operations
- **Block Lanczos** — Iterative Krylov subspace method operating on blocks of 64 vectors
- **Structured Gaussian Elimination** — Preprocessing to reduce matrix size before Block Lanczos
- **B-orthogonalization** — Proper three-term recurrence maintaining B-orthogonality between blocks
- **Automatic method selection** — Dense for small matrices (<500 relations), Block Lanczos for large

**Mathematical guarantees:**
- All nullspace vectors satisfy Ax = 0 over GF(2)
- Vectors are linearly independent
- Preprocessing preserves nullspace structure

**Results:**
- O(n²) vs O(n³) for dense Gaussian elimination
- 64 tests covering GF(2) arithmetic, rank computation, nullspace correctness, GNFS integration

**Files:** gnfs/linalg/sparse.py, gnfs/linalg/block_lanczos.py, tests/test_block_lanczos.py, tests/test_linalg_math.py`,
    },
    {
      date: '2026-01-30',
      title: 'Phase 1.2: Lattice Sieving',
      status: 'complete',
      content: `Completed lattice sieving implementation, a major improvement over the basic line sieve.

**What was implemented:**
- **Special-q sieving** — For each special prime q, sieve the sublattice L_q = {(a,b) : a ≡ rb (mod q)}
- **Lattice basis computation** — Natural basis v1=(q,0), v2=(r,1) with Lagrange reduction for shorter vectors
- **Smart special-q selection** — Filter for primes with polynomial roots to ensure productive sieving
- **Logarithmic sieving** — Efficient sieve using log approximations with trial factoring of candidates
- **Hybrid mode** — Automatic selection between line and lattice sieve based on factor base size

**Benchmark results (n=2021, 25 primes):**
- Line sieve: 1 relation
- Lattice sieve: 44 relations (44x more!)

**Why it matters:**
Lattice sieving reduces work by a factor of q for each special-q prime. By exploring multiple sublattices,
we find many more smooth relations in the same sieve region.

**Files:** gnfs/sieve/lattice_sieve.py, tests/test_lattice_sieve.py (30 tests)`,
    },
    {
      date: '2026-01-30',
      title: 'Phase 1.1: Polynomial Selection Improvements',
      status: 'complete',
      content: `Completed production-quality polynomial selection with major improvements over the naive approach.

**What was implemented:**
- **Murphy E scoring** — Alpha value computation, coefficient size scoring, root counting, smoothness analysis
- **Base-m expansion** — Express n in base m for mathematically correct GNFS polynomials where f(m) = n
- **Coefficient optimization** — Search around optimal m, leading coefficient adjustment, local optimization passes

**Results:**
- 6-12x better Murphy E scores for larger numbers
- Alpha values closer to 0 or negative (better small-prime divisibility)
- 32 new tests validating correctness

**Files changed:** gnfs/polynomial/selection.py, new test suite in tests/test_polynomial_selection.py`,
    },
    {
      date: '2026-01-30',
      title: 'Website Launch & Interactive Playground',
      status: 'complete',
      content: `Launched the educational website at gnfs-edu.web.app with:

- **Interactive playground** — Run GNFS in your browser via Pyodide
- **Detailed output** — Shows all four stages with actual calculations
- **Step-by-step learning** — Documentation for each phase of the algorithm

The playground successfully factors small semiprimes like 91 = 7 × 13 and 143 = 11 × 13, 
showing the polynomial selection, smooth relations found, and factor extraction.`,
    },
    {
      date: '2026-01-30',
      title: 'Production Roadmap Published',
      status: 'complete',
      content: `Published ROADMAP.md outlining the path from educational implementation to production-quality GNFS.

**Milestones:**
- v0.2: 40-digit numbers (improved polynomial selection)
- v0.3: 60-digit numbers (lattice sieving)
- v0.4: 80-digit numbers (Block Lanczos)
- v0.5: 100-digit numbers (multi-threading)
- v1.0: 120+ digit numbers (production ready)

**Key technical decisions:**
- Pure Python with optional gmpy2 for performance
- Target ~10x slower than CADO-NFS (acceptable for Python)`,
    },
  ]

  const planned = [
    {
      phase: '1.4',
      title: 'Square Root Improvements',
      description: "Montgomery's algorithm for efficient algebraic square roots in number fields.",
    },
    {
      phase: '2.1',
      title: 'Large Number Arithmetic',
      description: 'Integrate gmpy2 for GMP bindings with pure Python fallback.',
    },
    {
      phase: '2.2',
      title: 'Parallelization',
      description: 'Multi-threaded sieving and distributed linear algebra.',
    },
    {
      phase: '2.3',
      title: 'Memory Optimization',
      description: 'Streaming relations, memory-mapped files, compressed storage, checkpointing.',
    },
  ]

  return (
    <div className="min-h-screen pt-32 pb-24 px-8">
      <div className="max-w-[800px] mx-auto">
        <p className="section-label mb-3">Development</p>
        <h1 className="text-[clamp(36px,5vw,48px)] font-bold mb-4">Journal</h1>
        <p className="text-xl text-muted-foreground mb-12">
          Tracking our progress toward a production-quality GNFS implementation.
        </p>

        {/* Current Work */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold mb-6">Recent Updates</h2>
          <div className="space-y-6">
            {entries.map((entry, i) => (
              <article key={i} className="bg-card border border-border rounded-2xl p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="font-mono text-xs text-muted-foreground">{entry.date}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    entry.status === 'complete' 
                      ? 'bg-primary/20 text-primary' 
                      : 'bg-amber-500/20 text-amber-400'
                  }`}>
                    {entry.status === 'complete' ? '✓ Complete' : '◐ In Progress'}
                  </span>
                </div>
                <h3 className="text-lg font-semibold mb-3">{entry.title}</h3>
                <div className="text-sm text-muted-foreground whitespace-pre-line leading-relaxed">
                  {entry.content}
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Roadmap */}
        <section>
          <h2 className="text-2xl font-bold mb-6">Coming Up</h2>
          <div className="space-y-4">
            {planned.map((item, i) => (
              <div key={i} className="flex gap-4 p-4 bg-card/50 border border-border rounded-xl">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-muted flex items-center justify-center font-mono text-sm text-muted-foreground">
                  {item.phase}
                </div>
                <div>
                  <h3 className="font-semibold">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-6 text-sm text-muted-foreground">
            Full roadmap available on{' '}
            <a href="https://github.com/elliptic1/General-Number-Field-Sieve/blob/main/ROADMAP.md" 
               className="text-primary hover:underline">
              GitHub
            </a>
          </p>
        </section>
      </div>
    </div>
  )
}

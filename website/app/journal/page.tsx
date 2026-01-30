export default function JournalPage() {
  const entries = [
    {
      date: '2026-01-30',
      title: 'Phase 1.1: Polynomial Selection Improvements',
      status: 'in-progress',
      content: `Started work on production-quality polynomial selection. The current naive approach (f(x) = x - m) 
works for small numbers but produces poor polynomials that slow down sieving dramatically.

**What we're implementing:**
- Murphy E scoring to rate polynomial quality
- Base-m expansion for better polynomial construction  
- Coefficient optimization to find small, smooth coefficients
- Support for higher degree polynomials (4, 5, 6)

**Why it matters:**
Good polynomial selection can speed up the sieving phase by 10-100x. This is the single highest-impact 
improvement we can make.

**References:**
- Kleinjung (2006) "On polynomial selection for the general number field sieve"`,
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
      phase: '1.2',
      title: 'Lattice Sieving',
      description: 'Replace line sieve with lattice sieve and special-q. Exponentially faster for large numbers.',
    },
    {
      phase: '1.3', 
      title: 'Block Lanczos',
      description: 'Replace O(n³) Gaussian elimination with O(n²) Block Lanczos for sparse matrices.',
    },
    {
      phase: '1.4',
      title: 'Square Root Improvements',
      description: "Montgomery's algorithm for efficient algebraic square roots.",
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

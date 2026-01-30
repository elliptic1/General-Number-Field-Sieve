import Link from 'next/link'

const stages = [
  {
    number: 1,
    title: 'Polynomial Selection',
    description: 'Choose polynomials with shared roots modulo n. The foundation of the algebraic structure.',
    href: '/learn/polynomial',
  },
  {
    number: 2,
    title: 'Sieving',
    description: 'Find smooth relations using logarithmic sieve. The computational heart of GNFS.',
    href: '/learn/sieve',
  },
  {
    number: 3,
    title: 'Linear Algebra',
    description: 'Find dependencies using Gaussian elimination over GF(2). Sparse matrix techniques at scale.',
    href: '/learn/linear-algebra',
  },
  {
    number: 4,
    title: 'Square Root',
    description: 'Extract factors via congruence of squares. The algebraic payoff.',
    href: '/learn/square-root',
  },
]

const features = [
  {
    icon: '📖',
    title: 'Side-by-Side View',
    description: 'Read explanations alongside the actual Python source code that implements each concept.',
  },
  {
    icon: '⚡',
    title: 'Interactive Demos',
    description: 'Run real Python code in your browser. Experiment with different parameters and see results instantly.',
  },
  {
    icon: '🎯',
    title: 'Step by Step',
    description: 'Follow the algorithm through each stage with worked examples and visualizations.',
  },
]

export default function HomePage() {
  return (
    <>
      {/* Hero section */}
      <section className="min-h-screen flex items-center justify-center px-8 pt-24 pb-20 text-center">
        <div className="max-w-[800px]">
          <p className="font-mono text-xs text-primary uppercase tracking-[0.2em] mb-6">
            Integer Factorization · Number Theory · Cryptography
          </p>
          <h1 className="text-[clamp(40px,8vw,72px)] font-bold leading-[1.2] tracking-tight mb-6">
            The General Number <br />
            <span className="text-primary">Field Sieve</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-10 max-w-[600px] mx-auto">
            An interactive guide to the fastest known algorithm for factoring large integers.
            Learn the mathematics. Explore the code. Run live demonstrations.
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link href="/learn/introduction" className="btn-primary">
              Start Learning
            </Link>
            <Link 
              href="/playground" 
              className="inline-block px-8 py-4 text-sm font-semibold rounded-lg border border-border hover:border-primary transition-colors"
            >
              Try It Out
            </Link>
          </div>
        </div>
      </section>

      {/* Algorithm stages */}
      <section className="section-elevated py-24 px-8">
        <div className="max-w-[1200px] mx-auto">
          <div className="mb-12">
            <p className="section-label mb-3">The Algorithm</p>
            <h2 className="text-[clamp(32px,5vw,44px)] font-bold">Four Stages</h2>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stages.map((stage) => (
              <Link key={stage.number} href={stage.href}>
                <article className="h-full bg-card border border-border rounded-2xl p-8 transition-all duration-300 hover:border-primary/60 hover:-translate-y-1 card-glow">
                  <p className="category-label mb-3">Stage {stage.number}</p>
                  <h3 className="text-xl font-semibold mb-3">{stage.title}</h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {stage.description}
                  </p>
                </article>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Features section */}
      <section className="py-24 px-8">
        <div className="max-w-[1000px] mx-auto">
          <div className="text-center mb-12">
            <p className="section-label mb-3">Interactive Learning</p>
            <h2 className="text-[clamp(32px,5vw,44px)] font-bold">Learn by Doing</h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6">
            {features.map((feature) => (
              <div key={feature.title} className="text-center p-6">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Code preview section */}
      <section className="section-elevated py-24 px-8 border-t border-border">
        <div className="max-w-[800px] mx-auto text-center">
          <p className="section-label mb-3">Real Implementation</p>
          <h2 className="text-[clamp(28px,4vw,36px)] font-bold mb-4">
            Not a toy — a working factorization pipeline
          </h2>
          <p className="text-muted-foreground mb-8">
            Every stage mirrors the real algorithms behind GNFS. Polynomial selection, logarithmic sieving, 
            sparse matrix algebra, and algebraic square roots — all in readable Python.
          </p>
          
          <div className="bg-[hsl(220_20%_6%)] border border-border rounded-xl p-6 text-left overflow-x-auto">
            <pre className="font-mono text-sm text-muted-foreground">
              <code>{`from gnfs import gnfs_factor

# Factor a semiprime
n = 8051
factors = gnfs_factor(n)
print(f"{n} = {factors[0]} × {factors[1]}")
# → 8051 = 83 × 97`}</code>
            </pre>
          </div>
          
          <div className="mt-8">
            <Link href="/playground" className="font-mono text-sm text-primary hover:text-foreground transition-colors">
              Try it in the playground →
            </Link>
          </div>
        </div>
      </section>
    </>
  )
}

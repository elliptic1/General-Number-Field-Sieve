'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function FullPipelinePage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">The Complete GNFS Pipeline</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">Putting It All Together</h2>
        <p className="text-muted-foreground mb-4">
          Now that you understand each stage, let&apos;s see how they work together to factor
          an integer. The <code>gnfs_factor</code> function orchestrates the entire pipeline:
        </p>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">Pipeline Overview</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-sm space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">1</div>
              <div>
                <p className="font-semibold">Polynomial Selection</p>
                <p className="text-muted-foreground text-xs">select_polynomial(n, degree) &rarr; PolynomialSelection</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">2</div>
              <div>
                <p className="font-semibold">Sieving</p>
                <p className="text-muted-foreground text-xs">find_relations(selection, primes, interval) &rarr; [Relation, ...]</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">3</div>
              <div>
                <p className="font-semibold">Linear Algebra</p>
                <p className="text-muted-foreground text-xs">solve_matrix(relations, primes) &rarr; [[indices], ...]</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">4</div>
              <div>
                <p className="font-semibold">Square Root</p>
                <p className="text-muted-foreground text-xs">find_factors(n, relations, primes) &rarr; [p, q]</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <h2 className="text-xl font-semibold mt-8 mb-4">Parameters</h2>
        <p className="text-muted-foreground mb-4">
          The pipeline accepts several parameters that affect performance:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li><strong>degree</strong>: Polynomial degree (typically 1 for small n, higher for larger n)</li>
          <li><strong>bound</strong>: Smoothness bound B (primes up to B form the factor base)</li>
          <li><strong>interval</strong>: Sieve interval radius (how far to search for relations)</li>
          <li><strong>max_rounds</strong>: How many times to expand the interval if relations are sparse</li>
        </ul>

        <h2 className="text-xl font-semibold mt-8 mb-4">Automatic Interval Expansion</h2>
        <p className="text-muted-foreground mb-4">
          The implementation automatically expands the sieving interval if not enough relations
          are found. It needs at least len(primes) + 1 relations to guarantee the matrix has
          a non-trivial nullspace.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Example Run</h2>
        <Card className="mb-6">
          <CardContent className="font-mono text-sm pt-6">
            <p className="text-muted-foreground"># Factor 8051 using GNFS</p>
            <p>from gnfs import gnfs_factor</p>
            <p className="mt-2">factors = gnfs_factor(8051, bound=30, interval=50, degree=2)</p>
            <p>print(factors)  # [97, 83]</p>
            <p className="mt-4 text-muted-foreground"># Verify</p>
            <p>print(97 * 83)  # 8051 &check;</p>
          </CardContent>
        </Card>
      </section>

      {/* Demo placeholder */}
      <Card className="my-8 border-dashed">
        <CardHeader>
          <CardTitle className="text-base">Interactive Demo</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm mb-4">
            Run the complete GNFS pipeline on any integer. Watch each stage execute
            and see detailed output at every step.
          </p>
          <div className="bg-muted rounded-lg p-8 text-center text-muted-foreground">
            Demo loading... (Pyodide integration coming soon)
          </div>
        </CardContent>
      </Card>

      <div className="mt-8 flex justify-between">
        <Link href="/learn/square-root">
          <Button variant="outline">&larr; Square Root</Button>
        </Link>
        <Link href="/playground">
          <Button>Try the Playground &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

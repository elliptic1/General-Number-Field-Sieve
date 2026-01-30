'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function LinearAlgebraPage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Stage 3: Linear Algebra</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">The Goal</h2>
        <p className="text-muted-foreground mb-4">
          After sieving, we have many relations where both norms factor completely over our
          prime base. Now we need to find <strong>subsets</strong> of relations whose product
          is a perfect square on both sides.
        </p>
        <p className="text-muted-foreground mb-4">
          For a product to be a perfect square, every prime must appear an <strong>even</strong> number
          of times. This transforms our problem into linear algebra over GF(2) &mdash; the field
          with only 0 and 1 where 1 + 1 = 0.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Exponent Matrix</h2>
        <p className="text-muted-foreground mb-4">
          We build a matrix where:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li>Each <strong>column</strong> is a relation</li>
          <li>Each <strong>row</strong> is a prime in the factor base</li>
          <li>Entry (p, r) = exponent of prime p in relation r, mod 2</li>
        </ul>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">Example Matrix</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-sm">
            <p className="mb-2">Relations: r0, r1, r2, r3, r4</p>
            <p className="mb-2">Primes: 2, 3, 5, 7, 11</p>
            <pre className="bg-muted p-4 rounded text-xs overflow-x-auto">
{`       r0  r1  r2  r3  r4
   2 [  1   0   1   1   0 ]
   3 [  0   1   1   0   1 ]
   5 [  1   1   0   0   1 ]
   7 [  0   0   1   1   0 ]
  11 [  1   0   0   1   1 ]`}
            </pre>
          </CardContent>
        </Card>

        <h2 className="text-xl font-semibold mt-8 mb-4">Finding Dependencies</h2>
        <p className="text-muted-foreground mb-4">
          We want to find vectors v = (v₀, v₁, ...) in GF(2) such that Av = 0.
          This means the sum of the selected columns (where vᵢ = 1) has all zeros,
          i.e., every prime exponent sums to even.
        </p>
        <p className="text-muted-foreground mb-4">
          We use <strong>Gaussian elimination</strong> to compute the nullspace of the matrix.
          If we have more relations than primes, we&apos;re guaranteed to find at least one
          non-trivial dependency.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Algorithm</h2>
        <ol className="list-decimal list-inside space-y-2 text-muted-foreground mb-4">
          <li>Reduce the matrix to row echelon form using XOR operations</li>
          <li>Identify pivot columns and free columns</li>
          <li>For each free column, construct a nullspace basis vector</li>
          <li>Each basis vector tells us which relations to multiply together</li>
        </ol>
      </section>

      {/* Demo placeholder */}
      <Card className="my-8 border-dashed">
        <CardHeader>
          <CardTitle className="text-base">Interactive Demo</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm mb-4">
            Interactive matrix visualization will be loaded here.
            Step through Gaussian elimination and see dependencies form.
          </p>
          <div className="bg-muted rounded-lg p-8 text-center text-muted-foreground">
            Demo loading... (Pyodide integration coming soon)
          </div>
        </CardContent>
      </Card>

      <div className="mt-8 flex justify-between">
        <Link href="/learn/sieve">
          <Button variant="outline">&larr; Sieving</Button>
        </Link>
        <Link href="/learn/square-root">
          <Button>Next: Square Root &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

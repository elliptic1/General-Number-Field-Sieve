'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function SievePage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Stage 2: Sieving</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">What is Sieving?</h2>
        <p className="text-muted-foreground mb-4">
          The sieving stage is the computational heart of GNFS. Its goal is to find many
          pairs (a, b) such that both:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li>The <strong>algebraic norm</strong>: b<sup>d</sup>f(a/b) is B-smooth (only has prime factors &le; B)</li>
          <li>The <strong>rational norm</strong>: a - mb is also B-smooth</li>
        </ul>
        <p className="text-muted-foreground mb-4">
          These pairs are called <strong>relations</strong>. Once we have enough relations
          (more than the number of primes in our factor base), we can proceed to the
          linear algebra step.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Logarithmic Sieving</h2>
        <p className="text-muted-foreground mb-4">
          Rather than trial-dividing every candidate, we use a clever shortcut:
        </p>
        <ol className="list-decimal list-inside space-y-2 text-muted-foreground mb-4">
          <li>Initialize a sieve array with log(|value|) for each candidate</li>
          <li>For each prime p in the factor base, find where p divides the norm</li>
          <li>Subtract log(p) from those positions each time p divides</li>
          <li>After processing all primes, positions with values near zero are likely smooth</li>
        </ol>
        <p className="text-muted-foreground mb-4">
          This is much faster than trial division because subtraction is cheaper than division,
          and we only trial-factor the promising candidates.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Two-Sided Sieving</h2>
        <p className="text-muted-foreground mb-4">
          Real GNFS implementations sieve on both the algebraic and rational sides
          simultaneously. This filters out candidates that are only smooth on one side,
          dramatically reducing the number of trial factorizations needed.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Relation Object</h2>
        <p className="text-muted-foreground mb-4">
          Each successful relation stores:
        </p>
        <Card className="mb-6">
          <CardContent className="font-mono text-sm pt-6">
            <p>Relation(</p>
            <p className="pl-4">a = 11,              # First coordinate</p>
            <p className="pl-4">b = 7,               # Second coordinate (coprime to a)</p>
            <p className="pl-4">algebraic_value = 40,# b^d * f(a/b)</p>
            <p className="pl-4">rational_value = -23,# a - m*b</p>
            <p className="pl-4">algebraic_factors = &#123;2: 3, 5: 1&#125;,  # 2&sup3; &times; 5 = 40</p>
            <p className="pl-4">rational_factors = &#123;23: 1&#125;         # |-23| = 23</p>
            <p>)</p>
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
            Interactive sieve visualization will be loaded here.
            Watch the sieve array update as each prime is processed.
          </p>
          <div className="bg-muted rounded-lg p-8 text-center text-muted-foreground">
            Demo loading... (Pyodide integration coming soon)
          </div>
        </CardContent>
      </Card>

      <div className="mt-8 flex justify-between">
        <Link href="/learn/polynomial">
          <Button variant="outline">&larr; Polynomial Selection</Button>
        </Link>
        <Link href="/learn/linear-algebra">
          <Button>Next: Linear Algebra &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

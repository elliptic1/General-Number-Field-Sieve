'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { PolynomialDemo } from '@/components/demos/PolynomialDemo'

export default function PolynomialPage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Stage 1: Polynomial Selection</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">The Key Insight</h2>
        <p className="text-muted-foreground mb-4">
          GNFS works by operating in two different &ldquo;worlds&rdquo; simultaneously: the integers and
          an algebraic number field. The polynomial selection step creates the bridge between
          these worlds by choosing a polynomial f(x) such that:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li>f(x) has a root m modulo n (meaning f(m) &equiv; 0 mod n)</li>
          <li>f(x) is irreducible over the rationals</li>
          <li>The coefficients of f(x) are reasonably small</li>
        </ul>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Construction</h2>
        <p className="text-muted-foreground mb-4">
          This implementation uses the classic (x + m)<sup>d</sup> - n construction:
        </p>
        <ol className="list-decimal list-inside space-y-2 text-muted-foreground mb-4">
          <li>Choose the desired degree d</li>
          <li>Compute m = round(n<sup>1/d</sup>), the closest integer to the d-th root of n</li>
          <li>Expand (x + m)<sup>d</sup> and subtract n from the constant term</li>
        </ol>
        <p className="text-muted-foreground mb-4">
          This ensures that f(-m) = (-m + m)<sup>d</sup> - n = -n &equiv; 0 (mod n), so m is indeed
          a root of f(x) modulo n.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Example: n = 8051, degree = 2</h2>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">Worked Example</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-sm space-y-2">
            <p>n = 8051</p>
            <p>d = 2</p>
            <p>m = round(8051<sup>1/2</sup>) = round(89.7) = 90</p>
            <p className="mt-2">(x + 90)&sup2; = x&sup2; + 180x + 8100</p>
            <p>Subtract n: x&sup2; + 180x + 8100 - 8051 = x&sup2; + 180x + 49</p>
            <p className="mt-2 text-primary">f(x) = x&sup2; + 180x + 49</p>
            <p>Rational polynomial: g(x) = x - 90</p>
          </CardContent>
        </Card>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Code</h2>
        <p className="text-muted-foreground mb-4">
          The polynomial selection is implemented in <code>gnfs/polynomial/selection.py</code>.
          The key function is <code>select_polynomial(n, degree)</code> which returns a
          <code>PolynomialSelection</code> containing both the algebraic and rational polynomials.
        </p>
      </section>

      {/* Interactive Demo */}
      <div className="my-8">
        <h2 className="text-xl font-semibold mb-4">Try It Yourself</h2>
        <PolynomialDemo />
      </div>

      <div className="mt-8 flex justify-between">
        <Link href="/learn/introduction">
          <Button variant="outline">&larr; Introduction</Button>
        </Link>
        <Link href="/learn/sieve">
          <Button>Next: Sieving &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

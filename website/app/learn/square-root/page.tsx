'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function SquareRootPage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Stage 4: Square Root Step</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">The Final Step</h2>
        <p className="text-muted-foreground mb-4">
          The linear algebra stage gave us sets of relations whose exponents sum to even values.
          Now we combine these relations to form a <strong>congruence of squares</strong>:
        </p>
        <p className="text-center text-lg font-mono my-6">
          x&sup2; &equiv; y&sup2; (mod n)
        </p>
        <p className="text-muted-foreground mb-4">
          Once we have this, we can often factor n using:
        </p>
        <p className="text-center text-lg font-mono my-6">
          gcd(x - y, n)
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Computing x and y</h2>
        <p className="text-muted-foreground mb-4">
          For each dependency (set of relation indices), we compute:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li><strong>x</strong>: Product of all rational_values from selected relations (mod n)</li>
          <li><strong>y</strong>: Square root of the product of all algebraic_values</li>
        </ul>
        <p className="text-muted-foreground mb-4">
          Since the exponents sum to even values, the product of algebraic values is a perfect
          square, so we can compute its integer square root exactly.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Why Does This Work?</h2>
        <p className="text-muted-foreground mb-4">
          The key insight is the relationship between the algebraic and rational sides.
          For each relation (a, b):
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li>Algebraic side: b<sup>d</sup>f(a/b) factors over the algebraic number field</li>
          <li>Rational side: a - mb factors over the integers</li>
        </ul>
        <p className="text-muted-foreground mb-4">
          Because f(m) &equiv; 0 (mod n), these two sides are connected modulo n.
          When we multiply relations whose exponents sum to even values, both products
          become perfect squares that are congruent mod n.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Extracting Factors</h2>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-base">Example</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-sm space-y-2">
            <p>n = 8051</p>
            <p>After combining relations:</p>
            <p className="pl-4">x = 1234 (product of rational values mod n)</p>
            <p className="pl-4">y = 567 (sqrt of algebraic product)</p>
            <p className="mt-2">x&sup2; mod n = 1234&sup2; mod 8051 = 4567</p>
            <p>y&sup2; mod n = 567&sup2; mod 8051 = 4567 &check;</p>
            <p className="mt-2">gcd(1234 - 567, 8051) = gcd(667, 8051) = 97</p>
            <p className="text-primary mt-2">8051 = 97 &times; 83</p>
          </CardContent>
        </Card>

        <h2 className="text-xl font-semibold mt-8 mb-4">When It Fails</h2>
        <p className="text-muted-foreground mb-4">
          Sometimes gcd(x - y, n) gives a trivial factor (1 or n). This happens when
          x &equiv; &plusmn;y (mod n). In that case, we try another dependency from the
          nullspace basis. With enough relations, we almost always find a non-trivial factor.
        </p>
      </section>

      {/* Demo placeholder */}
      <Card className="my-8 border-dashed">
        <CardHeader>
          <CardTitle className="text-base">Interactive Demo</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm mb-4">
            Interactive square root demo will be loaded here.
            See how relations combine and factors emerge.
          </p>
          <div className="bg-muted rounded-lg p-8 text-center text-muted-foreground">
            Demo loading... (Pyodide integration coming soon)
          </div>
        </CardContent>
      </Card>

      <div className="mt-8 flex justify-between">
        <Link href="/learn/linear-algebra">
          <Button variant="outline">&larr; Linear Algebra</Button>
        </Link>
        <Link href="/learn/full-pipeline">
          <Button>Next: Full Pipeline &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

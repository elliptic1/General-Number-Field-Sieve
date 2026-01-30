import Link from 'next/link'
import { Button } from '@/components/ui/button'

export default function IntroductionPage() {
  return (
    <article className="p-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-4">Introduction to GNFS</h1>

      <section className="prose prose-slate dark:prose-invert max-w-none">
        <h2 className="text-xl font-semibold mt-8 mb-4">What is the General Number Field Sieve?</h2>
        <p className="text-muted-foreground mb-4">
          The General Number Field Sieve (GNFS) is the fastest known algorithm for factoring integers
          larger than about 100 digits. It was developed in the 1990s and has been used to factor
          increasingly large RSA challenge numbers, demonstrating the practical limits of cryptographic
          key sizes.
        </p>
        <p className="text-muted-foreground mb-4">
          Unlike simpler factoring methods like trial division or Pollard&apos;s rho, GNFS works by
          finding a &ldquo;congruence of squares&rdquo; &mdash; two numbers whose squares are congruent modulo n.
          Once we have x&sup2; &equiv; y&sup2; (mod n) with x &ne; &plusmn;y, we can often extract a factor
          using gcd(x - y, n).
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">The Four Stages</h2>
        <p className="text-muted-foreground mb-4">
          GNFS accomplishes this through four main stages:
        </p>
        <ol className="list-decimal list-inside space-y-2 text-muted-foreground mb-4">
          <li><strong>Polynomial Selection:</strong> Choose polynomials that share a common root modulo n</li>
          <li><strong>Sieving:</strong> Find pairs (a, b) where both algebraic and rational norms are smooth</li>
          <li><strong>Linear Algebra:</strong> Find combinations of relations whose exponents sum to even values</li>
          <li><strong>Square Root:</strong> Compute the actual square roots and extract factors via GCD</li>
        </ol>

        <h2 className="text-xl font-semibold mt-8 mb-4">Why Learn GNFS?</h2>
        <p className="text-muted-foreground mb-4">
          Understanding GNFS provides insight into:
        </p>
        <ul className="list-disc list-inside space-y-2 text-muted-foreground mb-4">
          <li>The mathematical foundations of modern cryptography</li>
          <li>Algebraic number theory and its practical applications</li>
          <li>Why RSA key sizes need to keep increasing</li>
          <li>Beautiful connections between abstract algebra and efficient computation</li>
        </ul>

        <h2 className="text-xl font-semibold mt-8 mb-4">About This Guide</h2>
        <p className="text-muted-foreground mb-4">
          This educational website lets you explore each stage of GNFS interactively. You&apos;ll see
          the actual Python code alongside explanations, and you can run live demonstrations
          right in your browser using Pyodide.
        </p>
        <p className="text-muted-foreground mb-4">
          The implementation is intentionally minimal and readable &mdash; production GNFS implementations
          have many optimizations that would obscure the core ideas. Our goal is to make the
          algorithm understandable, not to break RSA records.
        </p>
      </section>

      <div className="mt-8 flex justify-end">
        <Link href="/learn/polynomial">
          <Button>Next: Polynomial Selection &rarr;</Button>
        </Link>
      </div>
    </article>
  )
}

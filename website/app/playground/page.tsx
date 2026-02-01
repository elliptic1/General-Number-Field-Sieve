'use client'

import { useState } from 'react'
import { Slider } from '@/components/ui/slider'
import { usePyodide } from '@/hooks/usePyodide'

// Maximum number of digits allowed in playground (to prevent browser lockup)
const MAX_DIGITS = 20

export default function PlaygroundPage() {
  const [n, setN] = useState('91')
  const [degree, setDegree] = useState(1)
  const [bound, setBound] = useState(20)
  const [interval, setInterval] = useState(50)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const { isLoading, isReady, error: pyodideError, runCode, loadProgress } = usePyodide()

  const validateInput = (): string | null => {
    try {
      const num = BigInt(n)
      
      if (num < BigInt(2)) {
        return 'Number must be at least 2'
      }
      
      if (n.length > MAX_DIGITS) {
        return `Number too large for playground (max ${MAX_DIGITS} digits). Download the notebook to run locally.`
      }
    } catch (e) {
      return 'Invalid number format'
    }
    
    return null
  }

  const handleRun = async () => {
    if (!isReady) return
    
    // Validate input
    const validationError = validateInput()
    if (validationError) {
      setError(validationError)
      return
    }
    
    setIsRunning(true)
    setError(null)
    setResult(null)

    const pythonCode = `
import time
import sympy as sp
from gnfs import select_polynomial, find_relations, find_factors

n = ${n}
bound = ${bound}
interval = ${interval}
degree = ${degree}

start = time.time()

print("GNFS Factorization of n = " + str(n))
print("=" * 60)

# Primality check (polynomial time using Miller-Rabin)
print()
print("STEP 0: PRIMALITY CHECK")
print("-" * 60)
if sp.isprime(n):
    elapsed = time.time() - start
    print("Number is PRIME - cannot be factored!")
    print()
    print("=" * 60)
    print("RESULT: {} is prime".format(n))
    print("=" * 60)
    print("Time: {:.3f}s".format(elapsed))
    raise SystemExit(0)

# Step 1: Polynomial selection
print()
print("STEP 1: POLYNOMIAL SELECTION")
print("-" * 60)
selection = select_polynomial(n, degree=degree)
print("Choose m = floor(n^(1/d)) = " + str(selection.m))
print("Algebraic polynomial f(x) = " + str(selection.algebraic))
print("Rational polynomial g(x) = " + str(selection.rational))
print("These share root m = {} modulo n = {}".format(selection.m, n))

# Step 2: Build factor base and sieve
print()
print("STEP 2: SIEVING FOR SMOOTH RELATIONS")
print("-" * 60)
primes = list(sp.primerange(2, bound + 1))
print("Factor base (primes <= {}): {}".format(bound, primes))
print("Searching for relations in interval [-{}, {}]...".format(interval, interval))

relations = list(find_relations(selection, primes=primes, interval=interval))
print("Found {} smooth relations".format(len(relations)))

if relations:
    print()
    print("Relations (a, b) where a - {}*b factors over the base:".format(selection.m))
    shown = min(6, len(relations))
    for i, rel in enumerate(relations[:shown]):
        val = rel.a - selection.m * rel.b
        factors_str = " * ".join(
            "{}^{}".format(p, e) if e > 1 else str(p) 
            for p, e in rel.rational_factors.items()
        ) if rel.rational_factors else "1"
        print("  ({:4}, {:2}): {:4} - {}*{:2} = {:5} = {}".format(
            rel.a, rel.b, rel.a, selection.m, rel.b, val, factors_str
        ))
    if len(relations) > shown:
        print("  ... ({} more relations)".format(len(relations) - shown))

# Step 3: Linear algebra
print()
print("STEP 3: LINEAR ALGEBRA OVER GF(2)")
print("-" * 60)
print("Building exponent matrix ({} relations x {} primes)...".format(len(relations), len(primes)))
print("Finding vectors in nullspace (dependencies with even exponents)...")

# Step 4: Extract factors
print()
print("STEP 4: SQUARE ROOT AND FACTOR EXTRACTION")
print("-" * 60)

factors = list(find_factors(n, relations, primes))
elapsed = time.time() - start

if factors and len(factors) >= 2:
    p, q = factors[0], factors[1]
    print("Combining relations to form congruence of squares...")
    print("Found: x^2 = y^2 (mod n) where x != +/-y")
    print("gcd(x - y, n) gives non-trivial factor!")
    print()
    print("=" * 60)
    print("RESULT: {} = {} x {}".format(n, p, q))
    print("=" * 60)
    print("Verification: {} x {} = {}".format(p, q, p * q))
    print("Time: {:.3f}s".format(elapsed))
else:
    print("No factors found with current parameters.")
    print()
    print("Tips:")
    print("  - Try a smaller semiprime (91, 143 work well)")
    print("  - Increase smoothness bound")
    print("  - Increase sieve interval")
`

    try {
      const output = await runCode(pythonCode)
      setResult(output)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Execution failed')
    } finally {
      setIsRunning(false)
    }
  }

  const exampleNumbers = [
    { n: '91', note: '7 × 13', works: true },
    { n: '143', note: '11 × 13', works: true },
    { n: '221', note: '13 × 17', works: false },
    { n: '323', note: '17 × 19', works: false },
  ]

  return (
    <div className="min-h-screen pt-32 pb-24 px-8">
      <div className="max-w-[1100px] mx-auto">
        <p className="section-label mb-3">Interactive</p>
        <h1 className="text-[clamp(36px,5vw,48px)] font-bold mb-4">Playground</h1>
        <p className="text-xl text-muted-foreground mb-12 max-w-[600px]">
          Run the General Number Field Sieve in your browser. Real Python code, real factorization.
        </p>

        {isLoading && (
          <div className="mb-8 p-4 bg-card border border-border rounded-xl">
            <div className="flex items-center gap-3">
              <div className="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full" />
              <span className="text-muted-foreground">{loadProgress || 'Initializing Python runtime...'}</span>
            </div>
          </div>
        )}

        {pyodideError && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400">
            Failed to load Python: {pyodideError}
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-8">
          <div className="bg-card border border-border rounded-2xl p-8">
            <p className="category-label mb-2">Configuration</p>
            <h2 className="text-xl font-semibold mb-6">Parameters</h2>
            
            <div className="space-y-8">
              <div className="space-y-3">
                <label htmlFor="n" className="block text-sm font-medium">Integer to factor (n)</label>
                <input
                  id="n"
                  type="number"
                  value={n}
                  onChange={(e) => setN(e.target.value)}
                  placeholder="Enter an integer"
                  className="w-full px-4 py-3 bg-[hsl(var(--bg))] border border-border rounded-lg font-mono text-foreground focus:outline-none focus:border-primary transition-colors"
                />
                <p className="text-xs text-muted-foreground">
                  Small semiprimes work best (91, 143). Max {MAX_DIGITS} digits in playground - download notebook for larger numbers.
                </p>
              </div>

              <div className="space-y-3">
                <label className="block text-sm font-medium">Polynomial degree: <span className="font-mono text-primary">{degree}</span></label>
                <Slider value={degree} onChange={setDegree} min={1} max={4} step={1} />
                <p className="text-xs text-muted-foreground">
                  Degree 1 works for small numbers. Higher degrees for larger n.
                </p>
              </div>

              <div className="space-y-3">
                <label className="block text-sm font-medium">Smoothness bound: <span className="font-mono text-primary">{bound}</span></label>
                <Slider value={bound} onChange={setBound} min={10} max={100} step={5} />
                <p className="text-xs text-muted-foreground">
                  Primes up to this value form the factor base.
                </p>
              </div>

              <div className="space-y-3">
                <label className="block text-sm font-medium">Sieve interval: <span className="font-mono text-primary">{interval}</span></label>
                <Slider value={interval} onChange={setInterval} min={20} max={200} step={10} />
                <p className="text-xs text-muted-foreground">
                  Search range for smooth relations.
                </p>
              </div>

              <button
                onClick={handleRun}
                disabled={isRunning || !isReady || isLoading}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isLoading ? 'Loading Python...' : isRunning ? 'Factoring...' : 'Factor'}
              </button>

              <div className="pt-4 border-t border-border">
                <p className="text-sm font-medium mb-2">For Larger Numbers</p>
                <a
                  href="/gnfs_notebook.ipynb"
                  download
                  className="block w-full px-4 py-3 bg-[hsl(var(--bg))] border border-border rounded-lg text-center hover:border-primary transition-colors text-sm"
                >
                  Download Jupyter Notebook
                </a>
                <p className="text-xs text-muted-foreground mt-2">
                  Run on your own hardware with no size limits
                </p>
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-2xl p-8">
            <p className="category-label mb-2">Output</p>
            <h2 className="text-xl font-semibold mb-6">Results</h2>
            
            {error && (
              <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm font-mono">
                {error}
              </div>
            )}

            {result ? (
              <div className="bg-[hsl(var(--bg))] rounded-lg p-6 font-mono text-xs whitespace-pre-wrap text-muted-foreground border border-border overflow-x-auto max-h-[600px] overflow-y-auto">
                {result}
              </div>
            ) : (
              <div className="bg-[hsl(var(--bg))] rounded-lg p-12 text-center text-muted-foreground border border-border">
                {isReady 
                  ? 'Click "Factor" to run GNFS'
                  : 'Waiting for Python runtime to load...'}
              </div>
            )}
          </div>
        </div>

        <div className="mt-8 bg-card border border-border rounded-2xl p-8">
          <h3 className="font-semibold mb-4">Example Numbers</h3>
          <p className="text-muted-foreground text-sm mb-4">Green numbers work reliably with default parameters.</p>
          <div className="flex flex-wrap gap-3">
            {exampleNumbers.map((ex) => (
              <button
                key={ex.n}
                onClick={() => setN(ex.n)}
                className={`px-4 py-2 font-mono text-sm border rounded-lg transition-colors group ${
                  ex.works 
                    ? 'border-primary/50 hover:border-primary hover:text-primary' 
                    : 'border-border hover:border-muted-foreground'
                }`}
              >
                {ex.n}
                <span className="text-muted-foreground group-hover:text-primary/70 ml-2 text-xs">
                  ({ex.note})
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-8 bg-card border border-border rounded-2xl p-8">
          <h3 className="font-semibold mb-4">The Algorithm</h3>
          <div className="text-muted-foreground text-sm leading-relaxed space-y-3">
            <p><span className="text-foreground font-medium">Step 1 - Polynomial Selection:</span> Choose m = floor(n^(1/d)) and construct f(x) = x - m. Both f and g share m as a root mod n.</p>
            <p><span className="text-foreground font-medium">Step 2 - Sieving:</span> Find pairs (a,b) where a - mb factors completely over small primes (B-smooth). These are our relations.</p>
            <p><span className="text-foreground font-medium">Step 3 - Linear Algebra:</span> Build an exponent matrix over GF(2). Find dependencies where all exponents are even.</p>
            <p><span className="text-foreground font-medium">Step 4 - Square Root:</span> Combine relations to get x² ≡ y² (mod n). Then gcd(x-y, n) usually gives a factor.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

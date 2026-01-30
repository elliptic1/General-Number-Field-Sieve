const glossaryTerms = [
  {
    term: 'B-smooth',
    definition: 'An integer is B-smooth if all of its prime factors are less than or equal to B. Finding smooth numbers is the core task of the sieving step.',
  },
  {
    term: 'Factor Base',
    definition: 'The set of small primes (up to the smoothness bound B) used in sieving. Relations must factor completely over this set.',
  },
  {
    term: 'Relation',
    definition: 'A pair (a, b) with gcd(a, b) = 1 such that both the algebraic norm and rational norm are B-smooth. Each relation provides one equation in the linear algebra step.',
  },
  {
    term: 'Algebraic Norm',
    definition: 'For a relation (a, b), the algebraic norm is b^d * f(a/b) where f is the algebraic polynomial of degree d. This value lives in the algebraic number field.',
  },
  {
    term: 'Rational Norm',
    definition: 'For a relation (a, b), the rational norm is a - mb where m is the common root of the polynomials modulo n. This value is an ordinary integer.',
  },
  {
    term: 'GF(2)',
    definition: 'The finite field with two elements (0 and 1) where addition is XOR. Linear algebra over GF(2) finds combinations of relations with even total exponents.',
  },
  {
    term: 'Nullspace',
    definition: 'The set of vectors v such that Av = 0. In GNFS, nullspace vectors tell us which relations to combine to get a perfect square.',
  },
  {
    term: 'Congruence of Squares',
    definition: 'An equation x² ≡ y² (mod n) where x ≢ ±y (mod n). Such a congruence often reveals a factor: gcd(x - y, n) is usually non-trivial.',
  },
  {
    term: 'Polynomial Selection',
    definition: 'The first stage of GNFS: choosing polynomials f(x) and g(x) with a common root m modulo n. Good polynomial selection dramatically affects sieving efficiency.',
  },
  {
    term: 'Logarithmic Sieve',
    definition: 'An optimization where we work with logarithms of values. Instead of dividing by primes, we subtract log(p). Positions with small residual logs are likely smooth.',
  },
  {
    term: 'Gaussian Elimination',
    definition: 'A systematic method for solving linear systems by transforming the matrix to row echelon form. In GNFS, we use this over GF(2) to find dependencies.',
  },
  {
    term: 'Semiprime',
    definition: 'An integer that is the product of exactly two prime numbers. RSA moduli are semiprimes, making GNFS directly applicable to RSA cryptanalysis.',
  },
]

export default function GlossaryPage() {
  return (
    <div className="min-h-screen pt-32 pb-24 px-8">
      <div className="max-w-[800px] mx-auto">
        <p className="section-label mb-3">Reference</p>
        <h1 className="text-[clamp(36px,5vw,48px)] font-bold mb-4">Glossary</h1>
        <p className="text-xl text-muted-foreground mb-12">
          Key terms and concepts used in the General Number Field Sieve algorithm.
        </p>

        <div className="space-y-4">
          {glossaryTerms.map((item) => (
            <article key={item.term} className="bg-card border border-border rounded-2xl p-6 hover:border-primary/40 transition-colors">
              <h2 className="font-semibold text-lg mb-2 font-mono text-primary">{item.term}</h2>
              <p className="text-muted-foreground text-sm leading-relaxed">{item.definition}</p>
            </article>
          ))}
        </div>
      </div>
    </div>
  )
}

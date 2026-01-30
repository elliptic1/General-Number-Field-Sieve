import Link from 'next/link'

const sections = [
  {
    href: '/learn/introduction',
    title: 'Introduction',
    description: 'What is GNFS and why does it matter? A high-level overview of integer factorization.',
    stage: null,
  },
  {
    href: '/learn/polynomial',
    title: 'Polynomial Selection',
    description: 'How to choose polynomials that share a root modulo n, enabling the sieve to work.',
    stage: 1,
  },
  {
    href: '/learn/sieve',
    title: 'Sieving',
    description: 'Finding smooth relations using logarithmic sieving on both algebraic and rational sides.',
    stage: 2,
  },
  {
    href: '/learn/linear-algebra',
    title: 'Linear Algebra',
    description: 'Building an exponent matrix and finding its nullspace over GF(2) using Gaussian elimination.',
    stage: 3,
  },
  {
    href: '/learn/square-root',
    title: 'Square Root Step',
    description: 'Combining relations to form a congruence of squares and extracting factors via GCD.',
    stage: 4,
  },
  {
    href: '/learn/full-pipeline',
    title: 'Full Pipeline',
    description: 'See all four stages work together to factor an integer from start to finish.',
    stage: null,
  },
]

export default function LearnIndexPage() {
  return (
    <div className="min-h-screen pt-32 pb-24 px-8">
      <div className="max-w-[900px] mx-auto">
        <p className="section-label mb-3">Documentation</p>
        <h1 className="text-[clamp(36px,5vw,48px)] font-bold mb-4">Learn GNFS</h1>
        <p className="text-xl text-muted-foreground mb-12 max-w-[600px]">
          Work through each stage of the General Number Field Sieve algorithm with interactive examples.
        </p>

        <div className="grid gap-4">
          {sections.map((section) => (
            <Link key={section.href} href={section.href}>
              <article className="bg-card border border-border rounded-2xl p-8 transition-all duration-300 hover:border-primary/60 hover:-translate-y-0.5 flex gap-6 items-start">
                {section.stage && (
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold font-mono text-sm">
                    {section.stage}
                  </div>
                )}
                <div className={section.stage ? '' : 'pl-0'}>
                  <h2 className="text-lg font-semibold mb-2">{section.title}</h2>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {section.description}
                  </p>
                </div>
              </article>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}

'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'

const learnSections = [
  {
    title: 'Getting Started',
    items: [
      { href: '/learn/introduction', label: 'Introduction' },
    ],
  },
  {
    title: 'Algorithm Stages',
    items: [
      { href: '/learn/polynomial', label: '1. Polynomial Selection' },
      { href: '/learn/sieve', label: '2. Sieving' },
      { href: '/learn/linear-algebra', label: '3. Linear Algebra' },
      { href: '/learn/square-root', label: '4. Square Root' },
    ],
  },
  {
    title: 'Complete Pipeline',
    items: [
      { href: '/learn/full-pipeline', label: 'Full Factorization' },
    ],
  },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 shrink-0 border-r">
      <div className="sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto py-6 pr-6 pl-4">
        <nav className="space-y-6">
          {learnSections.map((section) => (
            <div key={section.title}>
              <h4 className="mb-2 text-sm font-semibold text-foreground">{section.title}</h4>
              <ul className="space-y-1">
                {section.items.map((item) => (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className={cn(
                        'block rounded-md px-3 py-2 text-sm transition-colors',
                        pathname === item.href
                          ? 'bg-accent text-accent-foreground font-medium'
                          : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      )}
                    >
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </div>
    </aside>
  )
}

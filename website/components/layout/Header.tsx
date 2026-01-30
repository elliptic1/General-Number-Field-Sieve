'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'
import { Menu, X, Github } from 'lucide-react'
import { useState, useEffect } from 'react'

const navItems = [
  { href: '/learn', label: 'Learn' },
  { href: '/playground', label: 'Playground' },
  { href: '/journal', label: 'Journal' },
  { href: '/reference/glossary', label: 'Glossary' },
]

export function Header() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <header className={cn(
      "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
      scrolled 
        ? "bg-[hsl(220_20%_4%/0.92)] backdrop-blur-xl py-3" 
        : "py-5"
    )}>
      <div className="max-w-[1200px] mx-auto px-8 flex justify-between items-center">
        <Link href="/" className="font-mono font-bold text-lg tracking-widest text-foreground">
          GNFS
        </Link>

        {/* Desktop navigation */}
        <nav className="hidden md:flex items-center gap-10">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "text-sm font-medium uppercase tracking-widest transition-colors",
                pathname?.startsWith(item.href) 
                  ? "text-foreground" 
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {item.label}
            </Link>
          ))}
          <a
            href="https://github.com/elliptic1/General-Number-Field-Sieve"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github className="h-5 w-5" />
          </a>
        </nav>

        {/* Mobile menu button */}
        <button
          className="md:hidden text-foreground p-2"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile navigation */}
      {mobileMenuOpen && (
        <nav className="md:hidden border-t border-border bg-[hsl(var(--bg-elevated))]">
          <div className="max-w-[1200px] mx-auto py-4 px-8 space-y-3">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "block py-2 text-sm font-medium uppercase tracking-widest transition-colors",
                  pathname?.startsWith(item.href) 
                    ? "text-foreground" 
                    : "text-muted-foreground hover:text-foreground"
                )}
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </nav>
      )}
    </header>
  )
}

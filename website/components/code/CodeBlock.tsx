'use client'

import { cn } from '@/lib/utils'

interface CodeBlockProps {
  code: string
  language?: string
  filename?: string
  highlightLines?: number[]
  className?: string
}

/**
 * Syntax-highlighted code block component.
 *
 * For now uses simple styling; can be enhanced with Shiki later.
 */
export function CodeBlock({
  code,
  language = 'python',
  filename,
  highlightLines = [],
  className,
}: CodeBlockProps) {
  const lines = code.split('\n')

  return (
    <div className={cn('rounded-lg border bg-muted overflow-hidden', className)}>
      {filename && (
        <div className="px-4 py-2 border-b bg-muted/50 text-sm text-muted-foreground font-mono">
          {filename}
        </div>
      )}
      <div className="overflow-x-auto code-scrollbar">
        <pre className="p-4 text-sm">
          <code className={`language-${language}`}>
            {lines.map((line, i) => (
              <div
                key={i}
                className={cn(
                  'px-2 -mx-2',
                  highlightLines.includes(i + 1) && 'bg-primary/10 border-l-2 border-primary'
                )}
              >
                <span className="inline-block w-8 text-muted-foreground text-right mr-4 select-none">
                  {i + 1}
                </span>
                {line || ' '}
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  )
}

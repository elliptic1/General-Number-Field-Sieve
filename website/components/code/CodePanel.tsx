'use client'

import { cn } from '@/lib/utils'
import { CodeBlock } from './CodeBlock'

interface CodePanelProps {
  children: React.ReactNode
  code: string
  filename?: string
  language?: string
  highlightLines?: number[]
  className?: string
}

/**
 * Side-by-side panel showing explanation and code.
 * On mobile, stacks vertically with explanation first.
 */
export function CodePanel({
  children,
  code,
  filename,
  language = 'python',
  highlightLines,
  className,
}: CodePanelProps) {
  return (
    <div className={cn('grid lg:grid-cols-2 gap-6', className)}>
      {/* Explanation side */}
      <div className="prose prose-slate dark:prose-invert max-w-none">
        {children}
      </div>

      {/* Code side */}
      <div className="lg:sticky lg:top-20 lg:self-start">
        <CodeBlock
          code={code}
          filename={filename}
          language={language}
          highlightLines={highlightLines}
        />
      </div>
    </div>
  )
}

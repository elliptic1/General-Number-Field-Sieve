'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { usePyodide } from '@/hooks/usePyodide'
import { Play, Loader2 } from 'lucide-react'

interface InteractiveCodeProps {
  initialCode: string
  className?: string
}

/**
 * Editable code block that can execute Python via Pyodide.
 */
export function InteractiveCode({ initialCode, className }: InteractiveCodeProps) {
  const [code, setCode] = useState(initialCode)
  const [output, setOutput] = useState<string | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const { isLoading, isReady, error, runCode, loadProgress } = usePyodide()

  const handleRun = async () => {
    if (!isReady) return

    setIsRunning(true)
    setOutput(null)

    try {
      const result = await runCode(code)
      setOutput(result || '(No output)')
    } catch (err) {
      setOutput(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className={cn('rounded-lg border bg-card overflow-hidden', className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/50">
        <span className="text-sm text-muted-foreground font-mono">Python</span>
        <Button
          size="sm"
          onClick={handleRun}
          disabled={isLoading || !isReady || isRunning}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              {loadProgress || 'Loading...'}
            </>
          ) : isRunning ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Run
            </>
          )}
        </Button>
      </div>

      {/* Code editor */}
      <div className="relative">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="w-full min-h-[200px] p-4 font-mono text-sm bg-background resize-y focus:outline-none"
          spellCheck={false}
        />
      </div>

      {/* Output */}
      {(output || error) && (
        <div className="border-t">
          <div className="px-4 py-2 bg-muted/50 text-sm text-muted-foreground">
            Output
          </div>
          <pre className="p-4 text-sm font-mono whitespace-pre-wrap overflow-x-auto bg-muted/30">
            {error ? (
              <span className="text-red-500">{error}</span>
            ) : (
              output
            )}
          </pre>
        </div>
      )}
    </div>
  )
}

'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { usePyodide } from '@/hooks/usePyodide'
import { Loader2 } from 'lucide-react'

interface PolynomialResult {
  m: number
  algebraic: string
  rational: string
  coefficients: number[]
}

export function PolynomialDemo() {
  const [n, setN] = useState('8051')
  const [degree, setDegree] = useState('2')
  const [result, setResult] = useState<PolynomialResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const { isLoading, isReady, error, runCode, loadProgress } = usePyodide()

  const handleRun = async () => {
    if (!isReady) return

    setIsRunning(true)
    setResult(null)

    const code = `
from gnfs.polynomial import select_polynomial

selection = select_polynomial(${n}, degree=${degree})
print(f"m = {selection.m}")
print(f"algebraic = {selection.algebraic}")
print(f"rational = {selection.rational}")
print(f"coefficients = {list(selection.algebraic.coeffs)}")
`

    try {
      const output = await runCode(code)
      // Parse output
      const lines = output.trim().split('\n')
      const m = parseInt(lines[0].split('=')[1].trim())
      const algebraic = lines[1].split('=')[1].trim()
      const rational = lines[2].split('=')[1].trim()
      const coeffsStr = lines[3].split('=')[1].trim()
      const coefficients = JSON.parse(coeffsStr.replace(/\(/g, '[').replace(/\)/g, ']'))

      setResult({ m, algebraic, rational, coefficients })
    } catch (err) {
      console.error('Error running polynomial selection:', err)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Polynomial Selection Demo</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="demo-n">Integer n</Label>
            <Input
              id="demo-n"
              type="number"
              value={n}
              onChange={(e) => setN(e.target.value)}
              placeholder="e.g., 8051"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="demo-degree">Degree</Label>
            <Input
              id="demo-degree"
              type="number"
              value={degree}
              onChange={(e) => setDegree(e.target.value)}
              min={1}
              max={5}
            />
          </div>
        </div>

        <Button
          onClick={handleRun}
          disabled={isLoading || !isReady || isRunning || !n || !degree}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              {loadProgress || 'Loading Pyodide...'}
            </>
          ) : isRunning ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Computing...
            </>
          ) : (
            'Select Polynomial'
          )}
        </Button>

        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-4 pt-4 border-t">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Root m:</span>
                <span className="ml-2 font-mono">{result.m}</span>
              </div>
              <div>
                <span className="text-muted-foreground">m&sup2;:</span>
                <span className="ml-2 font-mono">{result.m * result.m}</span>
              </div>
            </div>

            <div className="space-y-2">
              <div>
                <span className="text-muted-foreground text-sm">Algebraic polynomial f(x):</span>
                <div className="font-mono text-lg mt-1">{result.algebraic}</div>
              </div>
              <div>
                <span className="text-muted-foreground text-sm">Rational polynomial g(x):</span>
                <div className="font-mono text-lg mt-1">{result.rational}</div>
              </div>
            </div>

            <div className="p-4 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground mb-2">Verification:</p>
              <p className="font-mono text-sm">
                f(-m) = f(-{result.m}) &equiv; 0 (mod {n})
              </p>
              <p className="font-mono text-sm">
                g(-m) = -{result.m} - {result.m} = -{2 * result.m}
              </p>
            </div>
          </div>
        )}

        {!isReady && !isLoading && !error && (
          <div className="p-4 bg-muted rounded-lg text-center text-muted-foreground text-sm">
            Click the button to load Pyodide and run the demo
          </div>
        )}
      </CardContent>
    </Card>
  )
}

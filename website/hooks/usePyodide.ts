'use client'

import { useState, useCallback, useEffect } from 'react'
import { loadPyodideInstance, loadGNFSPackage, isPyodideLoaded } from '@/lib/pyodide'

interface UsePyodideResult {
  isLoading: boolean
  isReady: boolean
  error: string | null
  runCode: (code: string) => Promise<string>
  loadProgress: string
}

export function usePyodide(): UsePyodideResult {
  const [isLoading, setIsLoading] = useState(false)
  const [isReady, setIsReady] = useState(isPyodideLoaded())
  const [error, setError] = useState<string | null>(null)
  const [loadProgress, setLoadProgress] = useState('')

  useEffect(() => {
    if (isReady) return

    let mounted = true

    const init = async () => {
      setIsLoading(true)
      setLoadProgress('Loading Pyodide...')

      try {
        setLoadProgress('Downloading Pyodide runtime...')
        const pyodide = await loadPyodideInstance()

        if (!mounted) return

        setLoadProgress('Loading GNFS package...')
        await loadGNFSPackage(pyodide)

        if (!mounted) return

        setLoadProgress('')
        setIsReady(true)
        setError(null)
      } catch (err) {
        if (!mounted) return
        setError(err instanceof Error ? err.message : 'Failed to load Pyodide')
        setLoadProgress('')
      } finally {
        if (mounted) {
          setIsLoading(false)
        }
      }
    }

    init()

    return () => {
      mounted = false
    }
  }, [isReady])

  const runCode = useCallback(async (code: string): Promise<string> => {
    if (!isReady) {
      throw new Error('Pyodide is not ready')
    }

    const pyodide = await loadPyodideInstance()
    await loadGNFSPackage(pyodide)

    // Capture stdout/stderr
    const setupCode = `
import sys
from io import StringIO

_captured_output = StringIO()
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = _captured_output
sys.stderr = _captured_output
`
    
    const getOutputCode = `
sys.stdout = _old_stdout
sys.stderr = _old_stderr
_captured_output.getvalue()
`

    try {
      // Set up output capture
      await pyodide.runPythonAsync(setupCode)
      
      // Run the user's code - let errors bubble up naturally
      let userError: string | null = null
      try {
        await pyodide.runPythonAsync(code)
      } catch (err) {
        userError = err instanceof Error ? err.message : String(err)
      }
      
      // Get captured output
      const output = await pyodide.runPythonAsync(getOutputCode)
      const outputStr = String(output || '')
      
      // If there was an error, append it to the output
      if (userError) {
        return outputStr + '\n\nError: ' + userError
      }
      
      return outputStr
    } catch (err) {
      // Restore stdout/stderr if setup failed
      try {
        await pyodide.runPythonAsync(`
try:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
except:
    pass
`)
      } catch {
        // Ignore cleanup errors
      }
      throw new Error(err instanceof Error ? err.message : 'Python execution error')
    }
  }, [isReady])

  return {
    isLoading,
    isReady,
    error,
    runCode,
    loadProgress,
  }
}

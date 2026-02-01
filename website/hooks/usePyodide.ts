'use client'

import { useState, useCallback, useEffect, useRef } from 'react'

interface UsePyodideResult {
  isLoading: boolean
  isReady: boolean
  error: string | null
  runCode: (code: string) => Promise<string>
  loadProgress: string
}

export function usePyodide(): UsePyodideResult {
  const [isLoading, setIsLoading] = useState(true)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadProgress, setLoadProgress] = useState('Initializing...')
  
  const workerRef = useRef<Worker | null>(null)
  const pendingCallsRef = useRef<Map<number, {
    resolve: (value: string) => void
    reject: (error: Error) => void
  }>>(new Map())
  const nextIdRef = useRef(0)

  useEffect(() => {
    // Create worker
    const worker = new Worker('/pyodide.worker.js')
    workerRef.current = worker

    // Handle messages from worker
    worker.onmessage = (event) => {
      const { id, type, output, error: workerError, message } = event.data

      if (type === 'status') {
        setLoadProgress(message || '')
      } else if (type === 'ready') {
        setIsReady(true)
        setIsLoading(false)
        setLoadProgress('')
        setError(null)
      } else if (type === 'result') {
        const pending = pendingCallsRef.current.get(id)
        if (pending) {
          pending.resolve(output)
          pendingCallsRef.current.delete(id)
        }
      } else if (type === 'error') {
        if (id !== undefined) {
          const pending = pendingCallsRef.current.get(id)
          if (pending) {
            pending.reject(new Error(workerError))
            pendingCallsRef.current.delete(id)
          }
        } else {
          setError(workerError)
          setIsLoading(false)
        }
      }
    }

    worker.onerror = (err) => {
      setError(err.message || 'Worker error')
      setIsLoading(false)
    }

    // Initialize worker
    worker.postMessage({ type: 'init' })

    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  const runCode = useCallback(async (code: string): Promise<string> => {
    if (!isReady || !workerRef.current) {
      throw new Error('Pyodide is not ready')
    }

    return new Promise((resolve, reject) => {
      const id = nextIdRef.current++
      pendingCallsRef.current.set(id, { resolve, reject })
      
      workerRef.current!.postMessage({
        id,
        type: 'run',
        code,
      })
      
      // Timeout after 2 minutes
      setTimeout(() => {
        const pending = pendingCallsRef.current.get(id)
        if (pending) {
          pending.reject(new Error('Execution timeout (2 minutes)'))
          pendingCallsRef.current.delete(id)
        }
      }, 120000)
    })
  }, [isReady])

  return {
    isLoading,
    isReady,
    error,
    runCode,
    loadProgress,
  }
}

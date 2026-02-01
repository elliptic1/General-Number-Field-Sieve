// Web Worker for running Pyodide off the main thread
let pyodide = null
let pyodideReady = false

// Load Pyodide
async function loadPyodide() {
  if (pyodideReady) return pyodide
  
  self.postMessage({ type: 'status', message: 'Loading Pyodide runtime...' })
  
  // Load from CDN
  importScripts('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js')
  
  pyodide = await self.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
  })
  
  self.postMessage({ type: 'status', message: 'Installing packages...' })
  await pyodide.loadPackage(['numpy', 'sympy'])
  
  self.postMessage({ type: 'status', message: 'Loading GNFS module...' })
  
  // Fetch and load GNFS package files
  const gnfsFiles = [
    '/gnfs/__init__.py',
    '/gnfs/factor.py',
    '/gnfs/linalg/__init__.py',
    '/gnfs/linalg/matrix.py',
    '/gnfs/polynomial/__init__.py',
    '/gnfs/polynomial/number_field.py',
    '/gnfs/polynomial/polynomial.py',
    '/gnfs/polynomial/selection.py',
    '/gnfs/sieve/__init__.py',
    '/gnfs/sieve/lattice_sieve.py',
    '/gnfs/sieve/relation.py',
    '/gnfs/sieve/roots.py',
    '/gnfs/sieve/sieve.py',
    '/gnfs/sqrt/__init__.py',
    '/gnfs/sqrt/square_root.py',
  ]
  
  for (const file of gnfsFiles) {
    try {
      const response = await fetch(file)
      if (!response.ok) continue
      const content = await response.text()
      const path = file.substring(1) // Remove leading slash
      
      // Create directory structure
      const parts = path.split('/')
      let currentPath = ''
      for (let i = 0; i < parts.length - 1; i++) {
        currentPath += (currentPath ? '/' : '') + parts[i]
        try {
          pyodide.FS.mkdir(currentPath)
        } catch (e) {
          // Directory might already exist
        }
      }
      
      // Write file
      pyodide.FS.writeFile(path, content)
    } catch (e) {
      console.error(`Failed to load ${file}:`, e)
    }
  }
  
  pyodideReady = true
  self.postMessage({ type: 'ready' })
  return pyodide
}

// Handle messages from main thread
self.onmessage = async (event) => {
  const { id, type, code } = event.data
  
  try {
    if (type === 'init') {
      await loadPyodide()
      self.postMessage({ id, type: 'ready' })
      return
    }
    
    if (type === 'run') {
      if (!pyodideReady) {
        await loadPyodide()
      }
      
      // Set up output capture
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
        
        // Run user code
        let userError = null
        try {
          await pyodide.runPythonAsync(code)
        } catch (err) {
          userError = err.message || String(err)
        }
        
        // Get output
        const output = await pyodide.runPythonAsync(getOutputCode)
        const outputStr = String(output || '')
        
        if (userError) {
          self.postMessage({ 
            id, 
            type: 'result', 
            output: outputStr + '\n\nError: ' + userError 
          })
        } else {
          self.postMessage({ id, type: 'result', output: outputStr })
        }
      } catch (err) {
        // Restore stdout/stderr
        try {
          await pyodide.runPythonAsync(`
try:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
except:
    pass
`)
        } catch {}
        
        self.postMessage({ 
          id, 
          type: 'error', 
          error: err.message || String(err) 
        })
      }
    }
  } catch (err) {
    self.postMessage({ 
      id, 
      type: 'error', 
      error: err.message || String(err) 
    })
  }
}

// Start loading Pyodide immediately
loadPyodide().catch(err => {
  self.postMessage({ 
    type: 'error', 
    error: err.message || String(err) 
  })
})

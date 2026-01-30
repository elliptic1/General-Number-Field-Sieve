/**
 * Pyodide initialization and helper functions.
 *
 * Pyodide allows running Python in the browser via WebAssembly.
 * We use it to execute the actual GNFS Python code for interactive demos.
 */

// Type definitions for Pyodide
interface Pyodide {
  runPython: (code: string) => unknown;
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (packages: string | string[]) => Promise<void>;
  globals: Map<string, unknown>;
  FS: {
    writeFile: (path: string, data: string) => void;
    mkdir: (path: string) => void;
  };
}

declare global {
  interface Window {
    loadPyodide: (config: { indexURL: string }) => Promise<Pyodide>;
  }
}

let pyodideInstance: Pyodide | null = null;
let pyodideLoading: Promise<Pyodide> | null = null;

const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/';
const REQUIRED_PACKAGES = ['sympy', 'numpy'];

/**
 * Load and initialize Pyodide with required packages.
 * Returns cached instance if already loaded.
 */
export async function loadPyodideInstance(): Promise<Pyodide> {
  // Return cached instance
  if (pyodideInstance) {
    return pyodideInstance;
  }

  // Return in-progress loading promise
  if (pyodideLoading) {
    return pyodideLoading;
  }

  pyodideLoading = (async () => {
    // Load Pyodide script if not already present
    if (typeof window.loadPyodide === 'undefined') {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script');
        script.src = `${PYODIDE_CDN}pyodide.js`;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load Pyodide'));
        document.head.appendChild(script);
      });
    }

    // Initialize Pyodide
    const pyodide = await window.loadPyodide({
      indexURL: PYODIDE_CDN,
    });

    // Load required packages
    await pyodide.loadPackage(REQUIRED_PACKAGES);

    // Set up Python path for GNFS package
    // Files are at /gnfs/__init__.py, so we need / in path
    await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, '/')
    `);

    pyodideInstance = pyodide;
    return pyodide;
  })();

  return pyodideLoading;
}

/**
 * Load GNFS Python files into Pyodide's virtual filesystem.
 */
export async function loadGNFSPackage(pyodide: Pyodide): Promise<void> {
  // Fetch the Python files from public/gnfs/
  const files = [
    '__init__.py',
    'factor.py',
    'polynomial/__init__.py',
    'polynomial/polynomial.py',
    'polynomial/selection.py',
    'polynomial/number_field.py',
    'sieve/__init__.py',
    'sieve/sieve.py',
    'sieve/relation.py',
    'sieve/roots.py',
    'linalg/__init__.py',
    'linalg/matrix.py',
    'sqrt/__init__.py',
    'sqrt/square_root.py',
  ];

  // Create directory structure
  const dirs = ['/gnfs', '/gnfs/polynomial', '/gnfs/sieve', '/gnfs/linalg', '/gnfs/sqrt'];
  for (const dir of dirs) {
    try {
      pyodide.FS.mkdir(dir);
    } catch {
      // Directory may already exist
    }
  }

  // Fetch and write each file
  for (const file of files) {
    try {
      const response = await fetch(`/gnfs/${file}`);
      if (response.ok) {
        const content = await response.text();
        pyodide.FS.writeFile(`/gnfs/${file}`, content);
      }
    } catch (error) {
      console.warn(`Failed to load /gnfs/${file}:`, error);
    }
  }
}

/**
 * Execute Python code and return the result.
 */
export async function runPython(code: string): Promise<string> {
  const pyodide = await loadPyodideInstance();
  await loadGNFSPackage(pyodide);

  try {
    const result = await pyodide.runPythonAsync(code);
    return String(result);
  } catch (error) {
    throw new Error(`Python error: ${error}`);
  }
}

/**
 * Check if Pyodide is loaded.
 */
export function isPyodideLoaded(): boolean {
  return pyodideInstance !== null;
}

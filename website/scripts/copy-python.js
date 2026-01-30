/**
 * Copy Python source files to public/gnfs/ for Pyodide access.
 *
 * This script runs during build to make the GNFS Python package
 * available to Pyodide in the browser.
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const SRC = path.join(ROOT, 'gnfs');
const DEST = path.join(__dirname, '../public/gnfs');

function copyDir(src, dest) {
  // Create destination directory
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      // Skip __pycache__ directories
      if (entry.name === '__pycache__') continue;
      copyDir(srcPath, destPath);
    } else if (entry.name.endsWith('.py')) {
      fs.copyFileSync(srcPath, destPath);
      console.log(`Copied: ${path.relative(ROOT, srcPath)}`);
    }
  }
}

console.log('Copying Python source files to public/gnfs/...');
console.log(`Source: ${SRC}`);
console.log(`Destination: ${DEST}`);
console.log('');

try {
  // Clean destination
  if (fs.existsSync(DEST)) {
    fs.rmSync(DEST, { recursive: true });
  }

  copyDir(SRC, DEST);
  console.log('\nDone!');
} catch (error) {
  console.error('Error copying files:', error);
  process.exit(1);
}

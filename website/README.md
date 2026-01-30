# GNFS Educational Website

An interactive educational website explaining the General Number Field Sieve algorithm.

## Development

### Prerequisites

- Node.js 18+
- npm

### Setup

```bash
cd website
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Build

```bash
npm run build
```

This will:
1. Copy Python source files to `public/gnfs/` for Pyodide access
2. Build the Next.js static site

### Project Structure

```
website/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx         # Root layout with Header/Footer
│   ├── page.tsx           # Landing page
│   ├── learn/             # Educational content
│   │   ├── introduction/
│   │   ├── polynomial/
│   │   ├── sieve/
│   │   ├── linear-algebra/
│   │   ├── square-root/
│   │   └── full-pipeline/
│   ├── playground/        # Interactive experimentation
│   └── reference/         # Glossary
├── components/
│   ├── ui/               # Base UI components (shadcn/ui style)
│   ├── layout/           # Header, Sidebar, Footer
│   ├── code/             # CodeBlock, CodePanel, InteractiveCode
│   └── demos/            # Interactive demo components
├── lib/
│   ├── utils.ts          # Utility functions (cn)
│   └── pyodide.ts        # Pyodide initialization
├── hooks/
│   └── usePyodide.ts     # React hook for Python execution
└── public/
    └── gnfs/             # Python source (copied during build)
```

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Pyodide** - Python in the browser via WebAssembly
- **Lucide React** - Icons

## Adding New Content

### Learn Pages

Each learn page follows this structure:
1. Explanation content (prose)
2. Worked examples (Card components)
3. Interactive demo (Pyodide-powered)
4. Navigation buttons (previous/next)

### Interactive Demos

Demos use the `usePyodide` hook to execute Python:

```tsx
import { usePyodide } from '@/hooks/usePyodide'

function MyDemo() {
  const { isReady, runCode } = usePyodide()

  const handleRun = async () => {
    const result = await runCode(`
from gnfs import gnfs_factor
print(gnfs_factor(8051))
    `)
    console.log(result)
  }

  return <button onClick={handleRun}>Run</button>
}
```

## Deployment

The site is configured for static export. Deploy to any static host:

### Vercel (recommended)
```bash
npm run build
# Deploy via Vercel CLI or GitHub integration
```

### GitHub Pages
```bash
npm run build
# Output is in `out/` directory
```

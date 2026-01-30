<div align="center">

# General Number Field Sieve

**The fastest known algorithm for factoring large integers — implemented in Python**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Live Demo](https://gnfs-edu.web.app) · [Documentation](https://gnfs-edu.web.app/learn) · [Report Bug](https://github.com/elliptic1/General-Number-Field-Sieve/issues)

<img src="https://img.shields.io/github/stars/elliptic1/General-Number-Field-Sieve?style=social" alt="GitHub stars">

</div>

---

## ✨ Features

- 🔢 **Real GNFS Implementation** — Not a toy. Implements the actual algorithm used to factor RSA keys.
- 📚 **Educational Focus** — Clear, readable code with extensive documentation explaining the math.
- 🌐 **Interactive Playground** — [Try it in your browser](https://gnfs-edu.web.app/playground) with Pyodide.
- 📖 **Full Manuscript** — Ships with a book-length guide: *The General Number Field Sieve: From Theory to Practice*.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/elliptic1/General-Number-Field-Sieve.git
cd General-Number-Field-Sieve

# Install dependencies
pip install sympy numpy

# Factor a number
python cli.py 8051
# → 8051 = 83 × 97
```

## 📦 Installation

```bash
pip install sympy numpy
```

Or use the included virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 💡 Usage

### Command Line

```bash
# Basic usage
python cli.py 91

# With custom parameters
python cli.py 8051 --degree 1 --bound 50 --interval 100
```

### As a Library

```python
from gnfs import gnfs_factor

# Factor a semiprime
factors = gnfs_factor(8051, bound=50, interval=100)
print(f"8051 = {factors[0]} × {factors[1]}")
# → 8051 = 83 × 97
```

### Interactive (Browser)

Visit [gnfs-edu.web.app/playground](https://gnfs-edu.web.app/playground) to run GNFS directly in your browser — no installation required.

## 🏗️ How It Works

The General Number Field Sieve factors integers through four stages:

```
┌─────────────────────┐     ┌─────────────────────┐
│  1. Polynomial      │────▶│  2. Sieving         │
│     Selection       │     │                     │
│  Choose f(x), g(x)  │     │  Find B-smooth      │
│  with shared root   │     │  relations          │
└─────────────────────┘     └──────────┬──────────┘
                                       │
┌─────────────────────┐     ┌──────────▼──────────┐
│  4. Square Root     │◀────│  3. Linear Algebra  │
│                     │     │                     │
│  Extract factors    │     │  Gaussian elim      │
│  via gcd(x-y, n)    │     │  over GF(2)         │
└─────────────────────┘     └─────────────────────┘
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Polynomial Selection** | `gnfs/polynomial/` | Constructs polynomials sharing a root mod n |
| **Sieving** | `gnfs/sieve/` | Logarithmic sieve to find smooth relations |
| **Linear Algebra** | `gnfs/linalg/` | Finds dependencies using Gaussian elimination |
| **Square Root** | `gnfs/sqrt/` | Combines relations to extract factors |

## 📁 Project Structure

```
General-Number-Field-Sieve/
├── gnfs/
│   ├── __init__.py          # Public API
│   ├── factor.py            # Main factorization pipeline
│   ├── polynomial/          # Polynomial selection
│   ├── sieve/               # Relation finding
│   ├── linalg/              # Matrix operations over GF(2)
│   └── sqrt/                # Square root extraction
├── book/                    # Full manuscript
├── website/                 # Interactive demo site
├── tests/                   # Test suite
├── cli.py                   # Command-line interface
└── README.md
```

## 📖 Documentation

- **[Interactive Tutorial](https://gnfs-edu.web.app/learn)** — Step-by-step guide with live examples
- **[API Reference](https://gnfs-edu.web.app/reference/glossary)** — Glossary of terms and concepts
- **[Book](book/manuscript.md)** — *The General Number Field Sieve: From Theory to Practice*

## 🧪 Testing

```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📖 Documentation improvements
- 🧪 Additional test cases

Please feel free to submit a Pull Request.

## 📚 References

- Lenstra, A. K., & Lenstra, H. W. (1993). *The Development of the Number Field Sieve*
- Pomerance, C. (1996). *A Tale of Two Sieves*
- Buhler, J. P., Lenstra, H. W., & Pomerance, C. (1993). *Factoring integers with the number field sieve*

## ⚠️ Disclaimer

This is an **educational implementation**. While it implements the real GNFS algorithm, it is not optimized for factoring large integers. For serious cryptographic work, use established tools like [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) or [msieve](https://github.com/radii/msieve).

## 📄 License

MIT © [Todd B Smith](https://toddbsmith.com)

---

<div align="center">

**[Website](https://gnfs-edu.web.app)** · **[GitHub](https://github.com/elliptic1/General-Number-Field-Sieve)** · **[Report Issue](https://github.com/elliptic1/General-Number-Field-Sieve/issues)**

</div>

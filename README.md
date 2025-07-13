# General-Number-Field-Sieve

This repository provides a small demonstration of the General Number Field
Sieve (GNFS) algorithm written in Python.  While far from optimised it attempts
to follow the real steps of GNFS: polynomial selection, sieving, linear algebra
over GF(2) and the square root phase.  The sieving step performs a genuine line
sieve using ``sympy`` for helper number theoretic routines.

## Usage

The project exposes a simple command line interface.  After cloning the
repository install the ``sympy`` dependency and you can factor an integer as
follows:

```bash
pip install sympy
```

```bash
python cli.py 30
```

This will run the simplified GNFS pipeline and print the factors of the given
number.

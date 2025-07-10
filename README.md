# General-Number-Field-Sieve

This repository provides a toy implementation of the General Number Field
Sieve (GNFS) algorithm written in Python. The goal is to illustrate the overall
structure of the algorithm rather than provide a production-ready factorization
utility.  The sieving step now performs a genuine (albeit small scale) line
sieve using `sympy` for arithmetic helpers such as prime generation.

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

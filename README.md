# General-Number-Field-Sieve

This repository provides a compact yet faithful implementation of the General
Number Field Sieve (GNFS) algorithm written in Python.  The code mirrors the
real pipeline used in large scale integer factorisation projects and consists
of the following stages:

1. **Polynomial selection** – ``gnfs.polynomial`` constructs a polynomial of
   the form ``(x + m)^d - n`` so that ``x = -m`` is a root modulo ``n`` and the
   constant term is small.  This is the classic GNFS approach and replaces the
   earlier toy ``x^d - n`` construction.
2. **Sieving** – ``gnfs.sieve`` performs a logarithmic line sieve over a factor
   base.  For each prime in the base the roots of the polynomial modulo that
   prime are located and their logarithms are subtracted from a sieve array.
   Entries with small residuals are trial‑factored to collect ``B``‑smooth
   relations.
3. **Linear algebra** – ``gnfs.linalg`` builds an exponent matrix over GF(2)
   from the collected relations and computes dependencies between them using
   Gaussian elimination.
4. **Square root step** – ``gnfs.sqrt`` combines dependent relations to produce
   a congruence of squares and extracts a non‑trivial factor of ``n`` via a
   greatest common divisor.

While the implementation is intentionally minimalist, each component now
reflects the genuine algorithms behind GNFS rather than toy placeholders.

## Project structure

The repository mirrors the architecture of production GNFS codebases.  Each
stage of the algorithm lives in its own module to make the pipeline explicit:

* `gnfs/polynomial` – constructs polynomials of the form `(x + m)^d - n` with a
  real root modulo `n` and a small constant term.
* `gnfs/sieve` – carries out a logarithmic line sieve, locating roots modulo
  primes and gathering `B`‑smooth relations.
* `gnfs/linalg` – builds an exponent matrix over `GF(2)` and performs Gaussian
  elimination to determine dependencies.
* `gnfs/sqrt` – combines dependent relations into a congruence of squares and
  extracts non‑trivial factors via the greatest common divisor.
* `gnfs/factor` – a thin orchestration layer that links the stages into a
  working factorisation pipeline.

Although compact, these modules implement the real mathematics of the sieve and
illustrate the depth of the underlying algorithms.

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
number.  The CLI accepts a few optional arguments:

```bash
python cli.py 30 --degree 1 --bound 40 --interval 60
```

Values for ``degree``, ``bound`` and ``interval`` are loaded from
``default_config.json`` if not specified on the command line.

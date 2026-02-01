# Repository Guidelines

## Project Structure & Module Organization
- `gnfs/` holds the algorithm pipeline: `polynomial/` builds `(x + m)^d - n` polynomials, `sieve/` collects B-smooth relations, `linalg/` performs Gaussian elimination over GF(2), `sqrt/` extracts factors, and `factor.py` orchestrates the stages. Public imports are exposed via `gnfs/__init__.py`.
- `cli.py` is the entry point; defaults live in `default_config.json`.
- `tests/` mirrors the pipeline with `test_*` modules and a simple `conftest.py` that prepends the repo root to `sys.path`.
- `book/` contains the manuscript; `website/` is a Next.js site for docs/demos—keep Node tooling isolated from the Python workflow.

## Build, Test, and Development Commands
- Install Python deps locally (Python ≥3.8): `python -m pip install -e .`
- Run the CLI (example): `python cli.py 30 --degree 1 --bound 40 --interval 60`
- Execute the test suite: `python -m pytest tests -q`
- Website (optional): from `website/`, `npm install` (once) then `npm run dev` for local preview.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case for modules, files, functions, and variables.
- Prefer pure, side-effect-light functions; keep numeric routines deterministic and avoid implicit globals.
- Use docstrings (triple double quotes) and concise inline comments only where logic is non-obvious.
- Keep dependencies minimal (currently `sympy`, `numpy`); avoid adding heavy math libs without justification.

## Testing Guidelines
- Add `tests/test_<feature>.py` companions for new functionality; target small integers to keep runs fast.
- Prefer parametrized cases over long loops; assert both factorization output and intermediate relation counts where relevant.
- Run `python -m pytest tests` before pushing; include regression cases for any fixed bug.

## Commit & Pull Request Guidelines
- Match the existing history: short, imperative commit subjects (e.g., “Improve sieving and relation gathering”).
- In PRs, include: what changed, why it’s correct, and how to reproduce (CLI command with sample `n`). Link issues when applicable.
- Update docs (`README.md`, `book/`, or `website/`) when user-facing behavior or parameters change; note any config defaults touched in `default_config.json`.

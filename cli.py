"""Command-line interface for the minimal GNFS demonstration."""

import argparse
import json
from pathlib import Path
import sys

from gnfs import gnfs_factor

DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")


def load_config(path: Path) -> dict:
    """Load configuration options from ``path`` if it exists."""
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Factor integers using a minimal GNFS")
    parser.add_argument("n", type=int, help="Integer to factor")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to configuration file")
    parser.add_argument("--degree", type=int, help="Polynomial degree")
    parser.add_argument("--bound", type=int, help="Factor base bound")
    parser.add_argument("--interval", type=int, help="Sieve interval")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    degree = args.degree if args.degree is not None else cfg.get("degree", 1)
    bound = args.bound if args.bound is not None else cfg.get("bound", 30)
    interval = args.interval if args.interval is not None else cfg.get("interval", 50)

    factors = gnfs_factor(args.n, bound=bound, interval=interval, degree=degree)
    print(f"Factors of {args.n}: {factors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

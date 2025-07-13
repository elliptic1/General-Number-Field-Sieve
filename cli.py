"""Command-line interface for the minimal GNFS demonstration."""

import argparse
import sys

from gnfs import gnfs_factor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Factor integers using a minimal GNFS")
    parser.add_argument("n", type=int, help="Integer to factor")
    args = parser.parse_args(argv)

    factors = gnfs_factor(args.n)
    print(f"Factors of {args.n}: {factors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

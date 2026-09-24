"""Command-line entry point for the synthetic TN–DEC demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import Settings, run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="tn-dec-demo",
        description="Run the synthetic TN-DEC prediction and Pareto comparison workflow.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_outputs/run_01"),
        help="Destination directory; it must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Synthetic data generation seed.")
    parser.add_argument(
        "--rows", type=int, default=480, help="Number of synthetic daily records (180-2000)."
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=256,
        help="Surrogate evaluations per search (multiple of 32, 64-4096).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink to one scenario, one optimiser seed and 64 evaluations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the workflow and print a compact JSON summary to stdout."""
    args = parse_args(argv)
    settings = Settings(
        seed=args.seed,
        rows=args.rows,
        evaluations=64 if args.quick else args.evaluations,
        optimizer_seeds=(11,) if args.quick else (11, 23, 37),
        scenarios=1 if args.quick else 3,
    )
    result = run(settings, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "synthetic_only": True,
                "selected": result["selected"],
                "test": result["test"],
                "report": str(args.output / "report.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

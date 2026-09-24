"""Measure repeated synthetic demonstration runs across several data seeds.

This helper never loads private research inputs. It runs the public demonstration
once per requested seed and writes an aggregate ``benchmark_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Allow ``python tools/benchmark_demo.py`` to import the package without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tn_dec_demo.workflow import Settings, run

TARGETS = ("TN_out", "DEC")
METRICS = ("R2", "RMSE", "MAE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Repeat the synthetic demonstration across several dataset seeds."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_outputs/benchmark"),
        help="Destination directory; it must not already exist.",
    )
    parser.add_argument(
        "--seeds", default="17,29,43", help="Comma-separated, unique dataset seeds."
    )
    parser.add_argument(
        "--rows", type=int, default=480, help="Number of synthetic daily records per seed."
    )
    parser.add_argument(
        "--evaluations", type=int, default=256, help="Surrogate evaluations per search."
    )
    return parser.parse_args(argv)


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed test metrics into means and standard deviations."""
    targets: dict[str, Any] = {}
    for target in TARGETS:
        targets[target] = {
            metric: {
                "mean": statistics.mean(
                    run_result["test"][target][metric] for run_result in results
                ),
                "sd": statistics.stdev(run_result["test"][target][metric] for run_result in results)
                if len(results) > 1
                else None,
            }
            for metric in METRICS
        }
    return targets


def main(argv: list[str] | None = None) -> None:
    """Run every requested seed and write the aggregate benchmark summary."""
    args = parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Dataset seeds must be unique.")

    args.output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        result = run(
            Settings(seed=seed, rows=args.rows, evaluations=args.evaluations),
            args.output / f"seed_{seed}",
        )
        results.append(result)
        print(
            json.dumps(
                {"seed": seed, "status": result["status"], "seconds": result["runtime_seconds"]}
            ),
            flush=True,
        )

    summary = {
        "synthetic_only": True,
        "dataset_seeds": seeds,
        "successful_runs": len(results),
        "settings": {"rows": args.rows, "evaluations": args.evaluations},
        "targets": summarise(results),
    }
    (args.output / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

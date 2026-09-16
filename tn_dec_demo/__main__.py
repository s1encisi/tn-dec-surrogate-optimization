from pathlib import Path
import argparse
import json

from .workflow import Settings, run


def main():
    parser = argparse.ArgumentParser(description="Run the synthetic TN–DEC research workflow.")
    parser.add_argument("--output", type=Path, default=Path("demo_outputs/run_01"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rows", type=int, default=480)
    parser.add_argument("--evaluations", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    settings = Settings(seed=args.seed, rows=args.rows,
                        evaluations=64 if args.quick else args.evaluations,
                        optimizer_seeds=(11,) if args.quick else (11, 23, 37),
                        scenarios=1 if args.quick else 3)
    result = run(settings, args.output)
    print(json.dumps({"status": result["status"], "synthetic_only": True,
                      "selected": result["selected"], "test": result["test"],
                      "report": str(args.output/"report.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

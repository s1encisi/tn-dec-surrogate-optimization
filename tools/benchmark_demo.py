"""Measure repeated synthetic runs. Never load private research inputs."""
from pathlib import Path
import argparse
import json
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tn_dec_demo.workflow import Settings, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("demo_outputs/benchmark"))
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--rows", type=int, default=480)
    parser.add_argument("--evaluations", type=int, default=256)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Dataset seeds must be unique.")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in seeds:
        result = run(Settings(seed=seed, rows=args.rows, evaluations=args.evaluations),
                     args.output/f"seed_{seed}")
        results.append(result)
        print(json.dumps({"seed": seed, "status": result["status"],
                          "seconds": result["runtime_seconds"]}), flush=True)
    summary = {"synthetic_only": True, "dataset_seeds": seeds,
               "successful_runs": len(results), "settings": {"rows": args.rows, "evaluations": args.evaluations},
               "targets": {}}
    for target in ("TN_out", "DEC"):
        summary["targets"][target] = {
            metric: {"mean": statistics.mean(r["test"][target][metric] for r in results),
                     "sd": statistics.stdev(r["test"][target][metric] for r in results) if len(results)>1 else None}
            for metric in ("R2", "RMSE", "MAE")}
    (args.output/"benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

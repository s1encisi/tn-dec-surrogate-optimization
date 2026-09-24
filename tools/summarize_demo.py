"""Aggregate synthetic demo runs into the public ``docs/DEMO_RESULTS.md`` summary.

Only aggregate results of the synthetic demonstration are published. Private research
inputs, frozen models and manuscript numbers are never read or disclosed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

MEASUREMENT_DATE = "2026-09-16"
SEARCH_METHODS = ("NSGA2", "SPEA2", "MOEAD", "UniformRandom")
TARGETS = ("TN_out", "DEC")
METRICS = ("R2", "RMSE", "MAE")
CODE_FENCE = "```"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the public aggregate summary of synthetic demo runs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Benchmark directory written by tools/benchmark_demo.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/DEMO_RESULTS.md"),
        help="Destination Markdown file.",
    )
    return parser.parse_args(argv)


def load_runs(input_dir: Path) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    """Load per-seed run summaries, ordered by dataset seed.

    The returned protocol belongs to the first seed. Every seed pins the same
    dependency set, so the recorded package versions are identical.
    """
    summaries: dict[int, dict[str, Any]] = {}
    protocols: dict[int, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("seed_*/summary.json")):
        protocol = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8"))
        seed = protocol["settings"]["seed"]
        summaries[seed] = json.loads(path.read_text(encoding="utf-8"))
        protocols[seed] = protocol
    seeds = sorted(summaries)
    if not seeds:
        raise ValueError(f"No seed_*/summary.json found under {input_dir}.")
    return seeds, [summaries[seed] for seed in seeds], protocols[seeds[0]]


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    """Load every search record produced across all dataset seeds."""
    return [
        record
        for path in sorted(input_dir.glob("seed_*/pareto.json"))
        for record in json.loads(path.read_text(encoding="utf-8"))
    ]


def build_lines(
    aggregate: dict[str, Any],
    seeds: list[int],
    runs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    digest: str,
) -> list[str]:
    """Render the aggregate summary as Markdown lines."""
    packages = "，".join(f"{name} {version}" for name, version in protocol["packages"].items())
    evaluations = aggregate["settings"]["evaluations"]
    total_evaluations = sum(record["n_eval"] for record in records)

    lines = [
        "# 合成数据演示的实测结果",
        "",
        f"测量日期：{MEASUREMENT_DATE}。本页只汇总合成数据演示，不披露真实工厂数据或论文数值。",
        "生成公式和范围均为人为设定。以下精度不能用于宣称真实工厂预测性能或已实现的节能效果。",
        "",
        "## 设置",
        "",
        f"- Python {platform.python_version()}；单进程、模型n_jobs=1。",
        f"- 直接依赖版本：{packages}。",
        f"- 数据生成种子：{seeds}，每次{aggregate['settings']['rows']}条记录。",
        "- 每次训练/验证/留出划分为60%/20%/20%；仅按验证RMSE选择候选。",
        "- 每次3个合成背景、3个优化种子、4种搜索方法。",
        f"- 每次搜索{evaluations}次双目标代理评价，"
        f"共{len(records)}次搜索运行、{total_evaluations}次代理评价。",
        "",
        "## 留出集精度",
        "",
        "| 目标 | R² 均值 ± SD | RMSE 均值 ± SD | MAE 均值 ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = aggregate["targets"][target]
        cells = [f"{row[metric]['mean']:.4f} ± {row[metric]['sd']:.4f}" for metric in METRICS]
        lines.append(f"| {target} | {' | '.join(cells)} |")
    lines += [
        "",
        "SD基于3个数据生成种子。演示单位分别为mg/L和kWh/d。数据来自同一个人工生成机制，"
        "该重复用于检查流程稳定性，不等于跨工厂外部验证。",
        "",
        "## 每组数据的模型选择与时间",
        "",
        "| 数据种子 | TN模型 | DEC模型 | 流程秒数 |",
        "|---|---|---|---:|",
    ]
    for seed, run in zip(seeds, runs, strict=True):
        selected = run["selected"]
        lines.append(
            f"| {seed} | {selected['TN_out']} | {selected['DEC']} | {run['runtime_seconds']:.3f} |"
        )
    lines += [
        "",
        "流程时间包含生成、训练、选择、留出评价、解释与搜索，不包含绘图，也不包含解释器冷启动。",
        "候选模型因验证结果而不同，因此各数据种子的运行时间不相同。"
        "计时是当前本机执行结果，不承诺其他硬件上的耗时。",
        "",
        "## 同预算搜索的实测时间",
        "",
        "| 方法 | 运行数 | p50 秒 | p95 秒 |",
        "|---|---:|---:|---:|",
    ]
    for method in SEARCH_METHODS:
        seconds = [record["seconds"] for record in records if record["method"] == method]
        lines.append(
            f"| {method} | {len(seconds)} | {np.quantile(seconds, 0.5):.4f} "
            f"| {np.quantile(seconds, 0.95):.4f} |"
        )
    lines += [
        "",
        "以上每种方法各27次运行，包含不同数据种子、模型和情景；p50/p95仅作描述性汇总。",
        "",
        "## 搜索质量按数据种子报告",
        "",
        "| 数据种子 | NSGA-II HV | SPEA2 HV | MOEA/D HV | 随机参照 HV |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed, run in zip(seeds, runs, strict=True):
        by_method = {row["method"]: row["HV_seed_mean"] for row in run["search_summary"]}
        cells = " | ".join(f"{by_method[method]:.6f}" for method in SEARCH_METHODS)
        lines.append(f"| {seed} | {cells} |")
    lines += [
        "",
        "每个HV先按优化种子在三个情景内平均，再汇总三个优化种子；"
        "各数据种子的代理和归一化范围不同，不将跨数据种子的HV差异解释为方法提升。",
        "本次三组合成演示中SPEA2的平均HV均更高，保留实际结果，不预设NSGA-II必须胜出。",
        "",
        "## 可复现命令",
        "",
        f"{CODE_FENCE}powershell",
        "python tools/benchmark_demo.py --output demo_outputs/benchmark_new",
        "python tools/summarize_demo.py --input demo_outputs/benchmark_new "
        "--output docs/DEMO_RESULTS.md",
        CODE_FENCE,
        "",
        "默认输出必须为新目录。原始运行摘要、拆分和哈希清单留在本地demo_outputs；"
        "公开仓库只包含生成代码与本页聚合结果。",
        "",
        f"数据摘要SHA-256：{digest}。",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> None:
    """Entry point: read a benchmark directory and write the aggregate summary."""
    args = parse_args(argv)
    summary_path = args.input / "benchmark_summary.json"
    aggregate = json.loads(summary_path.read_text(encoding="utf-8"))
    if aggregate["synthetic_only"] is not True:
        raise ValueError("Refusing to summarise inputs that are not marked synthetic-only.")

    seeds, runs, protocol = load_runs(args.input)
    records = load_records(args.input)

    if len(runs) != aggregate["successful_runs"] or not all(run["synthetic_only"] for run in runs):
        raise ValueError("Per-seed summaries disagree with the aggregate manifest.")
    if set(seeds) != set(aggregate["dataset_seeds"]):
        raise ValueError("Dataset seeds differ between per-seed summaries and the manifest.")

    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    lines = build_lines(aggregate, seeds, runs, records, protocol, digest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # ``newline="\n"`` keeps the artifact identical on Windows and POSIX and matches
    # the ``eol=lf`` policy declared for ``/docs/*.md`` in .gitattributes.
    args.output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("Synthetic summary written:", args.output)


if __name__ == "__main__":
    main()

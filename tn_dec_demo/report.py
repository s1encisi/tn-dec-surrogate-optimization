"""Render a local Chinese report and a static Pareto figure from synthetic outputs.

The report is built only from the already-computed result dictionary, so it can never
introduce a performance claim that the run did not actually measure.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

COLORS = {"NSGA2": "#0072B2", "SPEA2": "#D55E00", "MOEAD": "#009E73", "UniformRandom": "#7A5195"}
MARKERS = {"NSGA2": "o", "SPEA2": "s", "MOEAD": "^", "UniformRandom": "x"}

PAGE_STYLE = (
    "body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;"
    "color:#17212b}"
    "h1,h2{line-height:1.25}h2{margin-top:36px}img{max-width:100%}a{color:#005b88}"
    "table{border-collapse:collapse;width:100%}"
    "th,td{border:1px solid #d3dce3;padding:10px 14px;text-align:left}"
    "th{background:#edf3f6}.table-wrap{overflow:auto}.scope{color:#4b5b68}"
)


def html_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """Render a two-dimensional sequence as an escaped HTML table."""
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def build_markdown(summary: dict[str, Any]) -> list[str]:
    """Render the Markdown report body from a completed run summary."""
    lines = [
        "# TN–DEC 合成数据演示报告",
        "",
        "本报告来自人为生成的数据，用于展示预测、解释和多目标比较流程，不代表真实工厂绩效。",
        "",
        "## 独立留出集预测结果",
        "",
        "| 目标 | 验证集选出的模型 | 留出集 R² | RMSE | MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for target, score in summary["test"].items():
        lines.append(
            f"| {target} | {summary['selected'][target]} | {score['R2']:.4f} | "
            f"{score['RMSE']:.4f} | {score['MAE']:.4f} |"
        )
    lines += [
        "",
        "TN_out使用mg/L、DEC使用kWh/d作为演示单位，所有生成式和参数均为人为设定。",
        "",
        "## 同预算多目标比较",
        "",
        "| 方法 | HV种子均值 | HV种子SD | p50秒/情景运行 | p95秒/情景运行 | 运行数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["search_summary"]:
        sd = "—" if row["HV_seed_sd"] is None else f"{row['HV_seed_sd']:.6f}"
        lines.append(
            f"| {row['method']} | {row['HV_seed_mean']:.6f} | {sd} | "
            f"{row['seconds_p50']:.4f} | {row['seconds_p95']:.4f} | {row['runs']} |"
        )
    lines += [
        "",
        "## 同域参照下的折衷候选",
        "",
        "| 情景日期 | ΔTN 同域 | ΔDEC 同域 | 原始输入在范围内 |",
        "|---|---:|---:|---|",
    ]
    for item in summary.get("solutions", []):
        delta = item["delta_same_domain"]
        lines.append(
            f"| {item['date']} | {delta[0]:+.4f} | {delta[1]:+.2f} | "
            f"{'是' if item['historical_in_domain'] else '否'} |"
        )
    lines += [
        "",
        "折衷候选来自预先指定的NSGA-II合并档案，参照由原输入投影到同一经验范围得到。"
        "这一选择用于展示参照定义，不表示NSGA-II在所有指标上最佳。",
        "",
        "HV先按种子在情景内平均，再汇总种子。计时包含单次搜索的代理计算与算法开销；"
        "p50/p95仅描述本次小样本本机运行，不代表线上服务延迟。",
        "",
        "![合成情景Pareto解集](pareto.png)",
        "",
        "## 可追溯信息",
        "",
        f"- 全流程耗时：{summary['runtime_seconds']:.3f}秒（不含本报告绘图）。",
        "- 特征插补与缩放在训练部分或内折拟合；模型族只按验证集选择。",
        "- 选择锁写入后统一计算留出集指标；留出集不参与参数选择。",
        "- 置换重要性使用开发阶段模型和验证集，表示模型关联。",
        "- 搜索范围来自开发数据近邻；候选不能直接解释为现场控制命令。",
        "- 运行参数、随机种子、拆分、选择锁和产物哈希保存在同目录。",
        "",
    ]
    return lines


def build_page(summary: dict[str, Any]) -> str:
    """Render the standalone HTML report from a completed run summary."""
    predictions = html_table(
        ["目标", "选出模型", "留出R²", "RMSE", "MAE"],
        [
            [
                target,
                summary["selected"][target],
                f"{metrics['R2']:.4f}",
                f"{metrics['RMSE']:.4f}",
                f"{metrics['MAE']:.4f}",
            ]
            for target, metrics in summary["test"].items()
        ],
    )
    search = html_table(
        ["方法", "平均HV", "p50秒", "p95秒", "运行数"],
        [
            [
                row["method"],
                f"{row['HV_seed_mean']:.6f}",
                f"{row['seconds_p50']:.4f}",
                f"{row['seconds_p95']:.4f}",
                row["runs"],
            ]
            for row in summary["search_summary"]
        ],
    )
    solutions = html_table(
        ["情景日期", "ΔTN 同域", "ΔDEC 同域", "历史输入在范围内"],
        [
            [
                item["date"],
                f"{item['delta_same_domain'][0]:+.4f}",
                f"{item['delta_same_domain'][1]:+.2f}",
                "是" if item["historical_in_domain"] else "否",
            ]
            for item in summary.get("solutions", [])
        ],
    )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>TN–DEC演示报告</title>"
        f"<style>{PAGE_STYLE}</style><body>"
        "<h1>水质与能耗的合成数据演示</h1>"
        '<p class="scope">完整流程：滞后输入 → 隔离验证 → 模型解释 → 同预算搜索 → 同域比较。</p>'
        "<p>数据来自人工生成机制，以下指标用于验证软件流程，不代表实际工厂性能或现场收益。</p>"
        f"<h2>留出集预测</h2>{predictions}<h2>同预算多目标比较</h2>{search}"
        '<p class="scope">HV按种子汇总；时间来自本机小样本实测，包含搜索与代理计算。</p>'
        "<h2>候选解集与同域参照</h2>"
        '<img src="pareto.png" alt="合成情景的Pareto候选、折衷点和同域参照">'
        f"{solutions}"
        '<p class="scope">NSGA-II被预先指定用于展示折衷候选；方法评价保留所有实际排名。'
        "经验范围不等于设备约束。</p>"
        '<h2>来源与复现</h2><p><a href="report.md">Markdown报告</a> · '
        '<a href="protocol.json">运行协议</a> · <a href="selection_lock.json">模型选择锁</a> · '
        '<a href="manifest.json">产物哈希清单</a></p>'
        "<p>留出数据不参与模型选择；置换重要性描述模型关联。完整输出用于离线复核。</p>"
        "</body></html>"
    )


def build_figure(summary: dict[str, Any], records: list[dict[str, Any]], output: Path) -> None:
    """Draw one Pareto panel per synthetic scenario and save it as ``pareto.png``."""
    dates = [case["date"] for case in summary["cases"]]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(
            1, len(dates), figsize=(5 * len(dates), 4), squeeze=False, layout="constrained"
        )
        for ax, date in zip(axes.flat, dates, strict=True):
            for method, color in COLORS.items():
                points = [
                    point
                    for row in records
                    if row["date"] == date and row["method"] == method
                    for point in row["front_f"]
                ]
                ax.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    s=14,
                    marker=MARKERS[method],
                    color=color,
                    alpha=0.65,
                    label=method,
                )
            ax.set(
                xlabel="Synthetic TN_out (mg/L)",
                ylabel="Synthetic DEC (kWh/d)",
                title=f"Synthetic scenario {date}",
            )
            item = next((s for s in summary.get("solutions", []) if s["date"] == date), None)
            if item:
                ax.scatter(
                    *item["compromise"], s=65, c="black", marker="*", label="Compromise", zorder=5
                )
                ax.scatter(
                    *item["same_domain_reference"],
                    s=42,
                    facecolors="none",
                    edgecolors="black",
                    marker="D",
                    label="Same-domain reference",
                    zorder=5,
                )
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.savefig(output / "pareto.png", dpi=160, facecolor="white")
        plt.close(fig)


def build_report(summary: dict[str, Any], records: list[dict[str, Any]], output: Path) -> None:
    """Write ``report.md``, ``pareto.png`` and ``report.html`` into ``output``."""
    markdown = "\n".join(build_markdown(summary))
    (output / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    build_figure(summary, records, output)
    (output / "report.html").write_text(build_page(summary), encoding="utf-8", newline="\n")

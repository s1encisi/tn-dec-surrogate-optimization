"""Create a public summary containing aggregate synthetic-demo results only."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("docs/DEMO_RESULTS.md"))
    args=parser.parse_args()
    aggregate=json.loads((args.input/"benchmark_summary.json").read_text(encoding="utf-8"))
    assert aggregate["synthetic_only"] is True
    runs_by_seed={}
    for p in args.input.glob("seed_*/summary.json"):
        protocol=json.loads((p.parent/"protocol.json").read_text(encoding="utf-8"))
        runs_by_seed[protocol["settings"]["seed"]]=json.loads(p.read_text(encoding="utf-8"))
    seeds=sorted(runs_by_seed);runs=[runs_by_seed[s] for s in seeds]
    records=[r for p in args.input.glob("seed_*/pareto.json")
             for r in json.loads(p.read_text(encoding="utf-8"))]
    assert len(runs)==aggregate["successful_runs"] and all(r["synthetic_only"] for r in runs)
    assert set(seeds)==set(aggregate["dataset_seeds"])
    lines=["# 合成数据演示的实测结果","",
           "测量日期：2026-09-16。本页只汇总合成数据演示，不披露真实工厂数据或论文数值。",
           "生成公式和范围均为人为设定。以下精度不能用于宣称真实工厂预测性能或已实现的节能效果。",
           "","## 设置","",
           f"- Python {platform.python_version()}；单进程、模型n_jobs=1。",
           "- 直接依赖版本："+ "，".join(f"{k} {v}" for k,v in protocol["packages"].items())+"。",
           f"- 数据生成种子：{seeds}，每次{aggregate['settings']['rows']}条记录。",
           "- 每次训练/验证/留出划分为60%/20%/20%；仅按验证RMSE选择候选。",
           "- 每次3个合成背景、3个优化种子、4种搜索方法。",
           f"- 每次搜索{aggregate['settings']['evaluations']}次双目标代理评价，共{len(records)}次搜索运行、{sum(r['n_eval'] for r in records)}次代理评价。",
           "","## 留出集精度","",
           "| 目标 | R² 均值 ± SD | RMSE 均值 ± SD | MAE 均值 ± SD |",
           "|---|---:|---:|---:|---:|"]
    for target in ["TN_out","DEC"]:
        row=aggregate["targets"][target]
        cells=[f"{row[m]['mean']:.4f} ± {row[m]['sd']:.4f}" for m in ["R2","RMSE","MAE"]]
        lines.append("| "+target+" | "+" | ".join(cells)+" |")
    lines += ["","SD基于3个数据生成种子。演示单位分别为mg/L和kWh/d。数据来自同一个人工生成机制，"
              "该重复用于检查流程稳定性，不等于跨工厂外部验证。",
              "","## 每组数据的模型选择与时间","",
              "| 数据种子 | TN模型 | DEC模型 | 流程秒数 |","|---|---|---|---:|"]
    for seed,r in zip(seeds,runs):
        lines.append(f"| {seed} | {r['selected']['TN_out']} | {r['selected']['DEC']} | {r['runtime_seconds']:.3f} |")
    lines += ["","流程时间包含生成、训练、选择、留出评价、解释与搜索，不包含绘图，也不包含解释器冷启动。",
              "候选模型因验证结果而不同，因此各数据种子的运行时间不相同。计时是当前本机执行结果，不承诺其他硬件上的耗时。",
              "","## 同预算搜索的实测时间","",
              "| 方法 | 运行数 | p50 秒 | p95 秒 |","|---|---:|---:|---:|"]
    for method in ["NSGA2","SPEA2","MOEAD","UniformRandom"]:
        values=[r["seconds"] for r in records if r["method"]==method]
        lines.append(f"| {method} | {len(values)} | {np.quantile(values,.5):.4f} | {np.quantile(values,.95):.4f} |")
    lines += ["","以上每种方法各27次运行，包含不同数据种子、模型和情景；p50/p95仅作描述性汇总。",
              "","## 搜索质量按数据种子报告","",
              "| 数据种子 | NSGA-II HV | SPEA2 HV | MOEA/D HV | 随机参照 HV |",
              "|---|---:|---:|---:|---:|"]
    for seed,r in zip(seeds,runs):
        mapping={x["method"]:x["HV_seed_mean"] for x in r["search_summary"]}
        lines.append("| "+str(seed)+" | "+" | ".join(f"{mapping[m]:.6f}" for m in ["NSGA2","SPEA2","MOEAD","UniformRandom"])+" |")
    lines += ["","每个HV先按优化种子在三个情景内平均，再汇总三个优化种子；"
              "各数据种子的代理和归一化范围不同，不将跨数据种子的HV差异解释为方法提升。",
              "本次三组合成演示中SPEA2的平均HV均更高，保留实际结果，不预设NSGA-II必须胜出。",
              "","## 可复现命令","",chr(96)*3+"powershell",
              "python tools/benchmark_demo.py --output demo_outputs/benchmark_new",
              "python tools/summarize_demo.py --input demo_outputs/benchmark_new --output docs/DEMO_RESULTS.md",
              chr(96)*3,"",
              "默认输出必须为新目录。原始运行摘要、拆分和哈希清单留在本地demo_outputs；"
              "公开仓库只包含生成代码与本页聚合结果。",
              "","数据摘要SHA-256："+hashlib.sha256((args.input/"benchmark_summary.json").read_bytes()).hexdigest()+"。",""]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("\n".join(lines),encoding="utf-8")
    print("Synthetic summary written:",args.output)

if __name__=="__main__":main()

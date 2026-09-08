"""Run the registered experiment into a fresh directory without replacing evidence."""
from pathlib import Path
import argparse
import runpy
import sys
parser=argparse.ArgumentParser()
parser.add_argument("--output-dir",required=True)
parser.add_argument("--phase",choices=["all","prepare","tune","evaluate","analyze"],default="all")
args=parser.parse_args()
out=Path(args.output_dir).resolve()
workspace=Path(__file__).resolve().parents[3]
if not out.is_relative_to(workspace):
    raise RuntimeError("Choose an output directory within this workspace")
if args.phase in ["all","prepare"] and out.exists():
    raise RuntimeError("Choose a fresh directory; existing evidence is never replaced")
import context_moo
context_moo.ROOT=out
import run_moo
if args.phase in ["all","prepare"]:run_moo.prepare()
if args.phase in ["all","tune"]:run_moo.run_tuning()
if args.phase in ["all","evaluate"]:run_moo.evaluate()
if args.phase in ["all","analyze"]:
    import pandas as pd
    frame=pd.read_csv(context_moo.PAPER/"results/prediction/final_outer_predictions.csv")
    for target,model in [("TN_out","Ensemble_Huber"),("DEC","ExtraTrees")]:
        part=frame[(frame.target==target)&(frame.training_window=="2023_2025")&(frame.model==model)].copy()
        part["residual"]=part.actual-part.prediction
        part.groupby("Date").residual.mean().rename(target).to_csv(out/"results"/f"{target}_unique_residuals.csv")
    runpy.run_path(str(Path(__file__).with_name("analyze_moo.py")),run_name="__main__")
    runpy.run_path(str(Path(__file__).with_name("refine_grid.py")),run_name="__main__")
    runpy.run_path(str(Path(__file__).with_name("anchor_diagnostics.py")),run_name="__main__")
    import json
    state_path=out/"results/run_status.json"
    state=json.loads(state_path.read_text(encoding="utf-8"))
    state.update(status="COMPLETE",analysis_summary_sha256=context_moo.sha(out/"results/analysis_summary.json"))
    context_moo.write_json(state_path,state)

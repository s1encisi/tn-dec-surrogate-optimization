"""Verify all saved Pareto outputs, budgets, source hashes, and selected solutions."""
from context_moo import *
from datetime import datetime, timezone
R=ROOT/"results"
p=json.loads((R/"protocol.json").read_text(encoding="utf-8"));lock=json.loads((R/"selection_lock.json").read_text(encoding="utf-8"))
for relative,expected in p["inputs"].items():
    assert sha(PAPER/relative)==expected,relative
assert sha(R/"protocol.json")==lock["protocol_sha256"]
assert sha(R/"tuning_summary.csv")==lock["tuning_summary_sha256"]
registry=pd.read_csv(R/"scenario_registry.csv").set_index("Date")
meta=json.loads((R/"context_contract.json").read_text(encoding="utf-8"));low=np.array(meta["objective_low"]);span=np.array(meta["objective_high"])-low
rows=[r for f in (R/"evaluation").glob("*.json") for r in json.loads(f.read_text(encoding="utf-8"))]
assert len(rows)==550
keys=[(r["date"],r["method"],r["seed"]) for r in rows];assert len(set(keys))==len(keys)
feature_parts=[];expected=[]
for row in rows:
    assert row["n_eval"]==2048
    assert row["seed"] in p["evaluation_seeds"]
    assert row["date"]>="2025-07-01"
    F=np.asarray(row["front_f"]);U=np.asarray(row["front_u"])
    assert len(nondominated_indices(F))==len(F)
    assert abs(hypervolume((F-low)/span)-row["hv"])<1e-12
    assert np.all(np.diff([h["hv"] for h in row["history"]])>=-1e-12)
    case=registry.loc[row["date"]].to_dict();case["Date"]=row["date"]
    feature_parts.append(feature_matrix(case,U));expected.append(F)
bundle=load_proxy()
observed=predict(bundle,pd.concat(feature_parts,ignore_index=True))
reference=np.vstack(expected);error=np.abs(observed-reference).max(axis=0)
assert error[0]<1e-7 and error[1]<1e-7,error
code_root=PAPER/"SourceCode";code_manifest=pd.read_csv(code_root/"CODE_MANIFEST.csv")
code_matches={r.relative_path:sha(code_root/r.relative_path)==r.copied_sha256 for _,r in code_manifest.iterrows()}
assert all(code_matches.values())
original_matches={path:sha(PAPER/path)==value for path,value in p["original_documents"].items()};assert all(original_matches.values())
frozen_time=pd.Timestamp(lock["created_utc"]).timestamp()
assert all(f.stat().st_mtime>=frozen_time for f in (R/"evaluation").glob("*.json"))
report=dict(status="PASS",checked_utc=datetime.now(timezone.utc).isoformat(),evaluation_runs=len(rows),
            proxy_evaluations=len(rows)*2048,pareto_points_recomputed=len(observed),max_error_TN=float(error[0]),max_error_DEC=float(error[1]),
            original_code_files_checked=len(code_matches),original_code_unchanged=True,original_documents=original_matches,
            protocol_inputs_match=True,selection_precedes_evaluation_files=True,all_budgets_exact=True,
            all_archives_nondominated=True,all_archive_hv_reconstructed=True,all_history_hv_monotonic=True)
write_json(R/"verification_report.json",report);print(json.dumps(report,ensure_ascii=False,indent=2))

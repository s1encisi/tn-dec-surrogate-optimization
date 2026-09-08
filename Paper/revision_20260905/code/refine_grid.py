"""Refine the grid on four calendar-selected contexts; no parameter changes."""
from context_moo import *
R=ROOT/"results"
summary=json.loads((R/"analysis_summary.json").read_text(encoding="utf-8"))
p=json.loads((R/"protocol.json").read_text(encoding="utf-8"))
meta=json.loads((R/"context_contract.json").read_text(encoding="utf-8"))
registry=pd.read_csv(R/"scenario_registry.csv").set_index("Date")
empirical=pd.read_csv(R/"empirical_reference_front.csv")
low=np.array(meta["objective_low"]);span=np.array(meta["objective_high"])-low
size=p["local_grid_refinement_size"];grid=np.linspace(0,1,size);U=np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2)
bundle=load_proxy();rows=[]
for date in summary["representative_dates"]:
    case=registry.loc[date].to_dict();case["Date"]=date
    F=predict(bundle,feature_matrix(case,U));keep=nondominated_indices(F)
    previous=empirical.loc[empirical.date==date,["TN_out","DEC"]].to_numpy()
    basehv=hypervolume((previous-low)/span);newhv=hypervolume((np.vstack([F[keep],previous])-low)/span)
    np.savez_compressed(R/"reference"/f"{date}_refined_surface.npz",U=U,F=F)
    old=json.loads((R/"reference"/f"{date}.json").read_text(encoding="utf-8"))
    rows.append(dict(date=date,grid_size=size*size,grid_hv_65=old["grid_hv"],grid_hv_129=hypervolume((F-low)/span),
                     empirical_hv_before=basehv,empirical_hv_after=newhv,relative_reference_hv_gain=(newhv-basehv)/max(basehv,1e-15)))
    print(json.dumps(rows[-1]),flush=True)
pd.DataFrame(rows).to_csv(R/"grid_refinement.csv",index=False)

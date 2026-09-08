import numpy as np
import pandas as pd
from context_moo import *

def test_dominance_and_hypervolume_against_known_geometry():
    f=np.array([[0.,1.],[1.,0.],[.5,.5],[.75,.75],[.5,.5]])
    assert set(nondominated_indices(f))=={0,1,2}
    assert np.isclose(hypervolume(f,(2,2)),3.25)
    assert igd_plus(f[:3],f[:3])==0
    assert igd_plus(f[:3],np.array([[2.,2.]]))>0

def test_support_contract_ignores_future_targets_and_future_fit_values():
    frame=load_initial_dataset(PAPER/"InitialData/initial_dataset.csv").frame
    first,contract=prepare_contexts(frame)
    changed=frame.copy(); mask=pd.to_datetime(changed.Date)>="2025-01-01"
    changed.loc[mask,["TN_out","DEC"]]=1e9
    second,contract2=prepare_contexts(changed)
    assert contract==contract2
    cols=["Date","selected","supported","low_PPA","high_PPA","low_DO","high_DO"]
    pd.testing.assert_frame_equal(first[cols],second[cols])
    changed.loc[mask,"T"]*=100
    _,contract3=prepare_contexts(changed)
    assert contract==contract3
    assert not set(first.loc[first.selected&(first.period=="tune"),"Date"])&set(first.loc[first.selected&(first.period=="evaluate"),"Date"])

def test_decision_mapping_preserves_context_and_smoothing_semantics():
    case={f:10. for f in TN_FEATURES}
    case.update(low_PPA=100,high_PPA=200,low_DO=1,high_DO=3)
    X=feature_matrix(case,np.array([[0,0],[1,1],[.5,.5]]))
    assert X.PPA.tolist()==[100,200,150]
    assert X.DO.tolist()==[1,3,2]
    assert X.MLSS.tolist()==[10,10,10]

def test_algorithms_budget_archive_and_interleaved_reproducibility():
    case={f:1. for f in TN_FEATURES};case.update(Date="2025-01-04",low_PPA=0,high_PPA=1,low_DO=0,high_DO=1)
    low=np.zeros(2);span=np.ones(2)
    for method in METHODS:
        cfg=configurations()[0]
        single=Job(case,method,cfg,11,128)
        while single.n<128:
            pop=single.ask();u=pop.get("X");f=np.column_stack([u[:,0],1-np.sqrt(u[:,0])+u[:,1]])
            single.tell(pop,f,low,span)
        first=Job(case,method,cfg,11,128);second=Job(case,method,cfg,23,128)
        while first.n<128:
            for job in (first,second):
                pop=job.ask();u=pop.get("X");f=np.column_stack([u[:,0],1-np.sqrt(u[:,0])+u[:,1]])
                job.tell(pop,f,low,span)
        assert single.n==128
        np.testing.assert_array_equal(np.vstack(single.all_u),np.vstack(first.all_u))
        np.testing.assert_array_equal(np.vstack(single.all_f),np.vstack(first.all_f))

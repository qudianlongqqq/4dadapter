#!/usr/bin/env python3
"""Predeclared 5K-step, 32-record head-only DeltaQ learnability diagnostic."""
from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap();REPORT=ROOT/'reports/ecir_mvr/sixs_deltaq_learnability_audit';ART=ROOT/'artifacts/ecir_mvr/sixs_deltaq_learnability_audit'; STATUS=REPORT/'HEAD_ONLY_STATUS.json'
import scripts.run_sixs_deltaq_prototype as run
from etflow.ecir.source_conditioned_deltaq import deltaq_targets

def aj(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n');os.replace(t,p)
def ac(p,f):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');f.to_csv(t,index=False);os.replace(t,p)
def corr(x,y):
    x=x.double().flatten();y=y.double().flatten();xc=x-x.mean();yc=y-y.mean();d=(xc.square().sum()*yc.square().sum()).sqrt();return float((xc*yc).sum()/d) if float(d)>0 else float('nan')
def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED')
    device=torch.device('cuda:0');prepared,payload=run.frozen.load_inputs();graphs,bg,source,reference,ids=run.fixed_overfit_batch(prepared,payload,device);tb,ta=deltaq_targets(source,reference,bg)
    model=run.build_model(device);groups=model.parameter_groups();trainable=list(groups['backbone'])+list(groups['deltaq']);allowed={id(x) for x in trainable}
    for p in model.parameters():p.requires_grad_(id(p) in allowed)
    opt=torch.optim.AdamW([{'params':groups['backbone'],'lr':1.5e-4},{'params':groups['deltaq'],'lr':3e-4}],weight_decay=1e-6);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5000)
    logs=[];model.train()
    with torch.no_grad():p=model.belief(bg,source,detach_sigma_features=True);ib=float((p['bond_deltaq']-tb).abs().mean());ia=float((p['angle_deltaq']-ta).abs().mean())
    aj(STATUS,{'status':'RUNNING','stage':'HEAD_ONLY_5000','pid':os.getpid(),'initial_bond_mae':ib,'initial_angle_mae':ia,'formal_test_read':'NO'})
    for step in range(1,5001):
        opt.zero_grad(set_to_none=True);p=model.belief(bg,source,detach_sigma_features=True);lb=(p['bond_deltaq']-tb).square().mean();la=(p['angle_deltaq']-ta).square().mean();loss=.5*(lb+la)
        if not torch.isfinite(loss):raise RuntimeError(f'NONFINITE_LOSS_STEP_{step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step();sched.step()
        if step==1 or step%25==0:
            row={'step':step,'loss':float(loss),'bond_mae':float((p['bond_deltaq']-tb).abs().mean()),'angle_mae':float((p['angle_deltaq']-ta).abs().mean()),'bond_signed_correlation':corr(p['bond_deltaq'],tb),'angle_signed_correlation':corr(p['angle_deltaq'],ta)};logs.append(row);ac(REPORT/'05_HEAD_ONLY_RESULTS.csv',pd.DataFrame(logs))
    model.eval()
    with torch.no_grad():p=model.belief(bg,source,detach_sigma_features=True);fb=float((p['bond_deltaq']-tb).abs().mean());fa=float((p['angle_deltaq']-ta).abs().mean());bc=corr(p['bond_deltaq'],tb);acorr=corr(p['angle_deltaq'],ta);finite=bool(torch.isfinite(p['bond_deltaq']).all() and torch.isfinite(p['angle_deltaq']).all())
    passed=finite and fb<=.1*ib and fa<=.1*ia and bc>=.9 and acorr>=.9
    ART.mkdir(parents=True,exist_ok=True);torch.save({'status':'PASS' if passed else 'FAIL','model_state':model.state_dict(),'step':5000,'initial_bond_mae':ib,'initial_angle_mae':ia,'final_bond_mae':fb,'final_angle_mae':fa,'bond_signed_correlation':bc,'angle_signed_correlation':acorr,'trained_parameter_groups':['backbone','deltaq'],'excluded':['sigma','Reliability','AdaptiveBA','tau','Cartesian action']},ART/'HEAD_ONLY_STEP5000.pt')
    aj(STATUS,{'status':'COMPLETED','stage':'HEAD_ONLY_COMPLETE','exit_code':0,'head_only_bond_final_mae':fb,'head_only_angle_final_mae':fa,'bond_signed_correlation':bc,'angle_signed_correlation':acorr,'finite_outputs':finite,'head_only_learnability':'PASS' if passed else 'FAIL','formal_test_read':'NO'})
if __name__=='__main__':main()

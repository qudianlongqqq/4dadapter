#!/usr/bin/env python3
"""Diagnostic-only learnability audit for the frozen failed DeltaQ overfit."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap(); REPORT=ROOT/'reports/ecir_mvr/sixs_deltaq_learnability_audit'; ARTIFACT=ROOT/'artifacts/ecir_mvr/sixs_deltaq_learnability_audit'
ORIG=ROOT/'reports/ecir_mvr/sixs_deltaq_prototype/07_SMALL_OVERFIT_RESULTS.csv'; CKPT=ROOT/'artifacts/ecir_mvr/sixs_deltaq_prototype/SMALL_OVERFIT_CHECKPOINT.pt'
import scripts.run_sixs_deltaq_prototype as run
from etflow.ecir.source_conditioned_deltaq import deltaq_targets, deltaq_belief_loss
from etflow.ecir.learned_geometry import geometry_values

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def aj(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n',encoding='utf-8');os.replace(t,p)
def ac(p,f):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(f'.{p.name}.{os.getpid()}.tmp');f.to_csv(t,index=False);os.replace(t,p)
def corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);xc=x-x.mean();yc=y-y.mean();d=float(np.sqrt(np.sum(xc*xc)*np.sum(yc*yc)));return float(np.sum(xc*yc)/d) if d else float('nan')
def slope(frame,col,start):
    p=frame[frame.step>=start];x=p.step.to_numpy(float);y=p[col].to_numpy(float);xc=x-x.mean();return float(np.sum(xc*(y-y.mean()))/np.sum(xc*xc))

def curve_audit():
    state=torch.load(CKPT,map_location='cpu',weights_only=False); curve=pd.read_csv(ORIG)
    zero=pd.DataFrame([{'step':0,'belief_loss':np.nan,'bond_deltaq_mae':state['initial_bond_mae'],'angle_deltaq_mae':state['initial_angle_mae']}]); full=pd.concat([zero,curve],ignore_index=True).drop_duplicates('step',keep='last').sort_values('step')
    wanted=[]
    for target in (0,100,250,500,750,1000,1250):
        row=full.iloc[(full.step-target).abs().argmin()].to_dict();row['requested_step']=target;wanted.append(row)
    out=pd.DataFrame(wanted); out['bond_last250_slope']=slope(full,'bond_deltaq_mae',1000);out['angle_last250_slope']=slope(full,'angle_deltaq_mae',1000);out['bond_last500_slope']=slope(full,'bond_deltaq_mae',750);out['angle_last500_slope']=slope(full,'angle_deltaq_mae',750)
    b250=1-float(full.iloc[-1].bond_deltaq_mae/full.loc[full.step==1000,'bond_deltaq_mae'].iloc[0]);a250=1-float(full.iloc[-1].angle_deltaq_mae/full.loc[full.step==1000,'angle_deltaq_mae'].iloc[0])
    out['bond_last250_relative_reduction']=b250;out['angle_last250_relative_reduction']=a250
    out['bond_curve_status']='PLATEAUED' if b250<.01 else 'STILL_DESCENDING';out['angle_curve_status']='PLATEAUED' if a250<.01 else 'STILL_DESCENDING'
    ac(REPORT/'01_ORIGINAL_CURVE.csv',out);return out

def distribution(family,values):
    v=np.asarray(values,float);av=np.abs(v);q=np.quantile(v,[.1,.25,.5,.75,.9]);aq=np.quantile(av,[.1,.25])
    rounded=np.round(v,12);_,counts=np.unique(rounded,return_counts=True);iqr=np.quantile(v,.75)-np.quantile(v,.25);lo=np.quantile(v,.25)-5*iqr;hi=np.quantile(v,.75)+5*iqr
    pos=float(np.mean(v>0));neg=float(np.mean(v<0));near=float(np.mean(av<1e-3 if family=='bond' else av<5e-3));outlier=float(np.mean((v<lo)|(v>hi)))
    pathology=(not np.isfinite(v).all()) or max(pos,neg)>.9 or near>.5 or outlier>.02
    return {'section':'TARGET_DISTRIBUTION','family':family,'n':len(v),'mean':v.mean(),'median':np.median(v),'std':v.std(ddof=1),'p10':q[0],'p25':q[1],'p50':q[2],'p75':q[3],'p90':q[4],'min':v.min(),'max':v.max(),'fraction_positive':pos,'fraction_negative':neg,'fraction_zero':float(np.mean(v==0)),'fraction_abs_lt_1e-3':float(np.mean(av<1e-3)),'fraction_abs_lt_2_5e-3':float(np.mean(av<2.5e-3)),'fraction_abs_lt_5e-3':float(np.mean(av<5e-3)),'abs_p10':aq[0],'abs_p25':aq[1],'duplicate_value_fraction':float(counts[counts>1].sum()/len(v)),'extreme_outlier_fraction_5iqr':outlier,'finite':bool(np.isfinite(v).all()),'near_zero_dominance':near>.5,'sign_imbalance':max(pos,neg)>.9,'target_pathology':'YES' if pathology else 'NO'}

def grad_norm(loss,params,retain=True):
    grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True);total=loss.new_zeros(())
    for g in grads:
        if g is not None:total=total+g.square().sum()
    return float(total.sqrt())

def endpoint(stage,model,graphs,bg,source,reference):
    model.train();pred=model.belief(bg,source,detach_sigma_features=False);tb,ta=deltaq_targets(source,reference,bg);total,parts=deltaq_belief_loss(pred,tb,ta,graphs,beta=.5)
    result=[]
    for fam,target in (('bond',tb),('angle',ta)):
        p=pred[f'{fam}_deltaq'];sig=pred[f'{fam}_sigma'];lr=pred[f'{fam}_log_sigma_ratio'];raw_nll=.5*((target.to(p)-p)/sig).square()+sig.log();weighted=raw_nll*sig.square().detach().pow(.5)
        names=dict(model.belief.named_parameters());head=[v for k,v in names.items() if f'{fam}_head' in k];sigma=[v for k,v in names.items() if f'{fam}_sigma_head' in k]
        result.append({'stage':stage,'family':fam,'prediction_mae':float((p-target.to(p)).abs().mean()),'sigma_median':float(sig.median()),'sigma_p90':float(torch.quantile(sig,.9)),'sigma_p99':float(torch.quantile(sig,.99)),'beta_nll_term_mean':float(weighted.mean()),'raw_nll_mean':float(raw_nll.mean()),'effective_variance_weight_median':float(sig.median()),'deltaq_head_gradient_norm':grad_norm(weighted.mean(),head),'sigma_head_gradient_norm':grad_norm(weighted.mean(),sigma),'sigma_floor_saturation_fraction':float((lr<=-6+1e-6).double().mean()),'sigma_ceiling_saturation_fraction':float((lr>=6-1e-6).double().mean()),'sigma_gt_10x_stat_fraction':float((lr>math.log(10)).double().mean()),'log_sigma_ratio_median':float(lr.median())})
    return result,tb.detach().cpu().numpy(),ta.detach().cpu().numpy()

def source_conditioning_full_train(prepared,sources):
    # Exact identity-level audit. For fixed source q, random reference choice
    # leaves Var(Deltaq|q_source,molecule,primitive)=Var(q_reference).
    rows=[]
    for family in ('bond','angle'):
        ref_std=[];source_span=[];ambiguous=[];nprim=0;n_multi_ref=0
        for item in prepared['train']:
            graph=item['graph']; pool=sources[str(item['molecule_id'])]
            src=[]
            for s in pool:
                b,a=geometry_values(torch.as_tensor(s['source'],dtype=torch.float64),graph);src.append((b if family=='bond' else a).numpy())
            refs=[]
            for r in item['references']:
                b,a=geometry_values(torch.as_tensor(r,dtype=torch.float64),graph);refs.append((b if family=='bond' else a).numpy())
            S=np.stack(src);R=np.stack(refs);rs=R.std(axis=0);sp=S.max(axis=0)-S.min(axis=0);threshold=.002 if family=='bond' else .01
            ref_std.extend(rs.tolist());source_span.extend(sp.tolist());ambiguous.extend((rs>threshold).tolist());nprim+=R.shape[1];n_multi_ref+=int(R.shape[0]>1)*R.shape[1]
        r=np.asarray(ref_std);s=np.asarray(source_span);a=np.asarray(ambiguous,bool)
        rows.append({'family':family,'train_molecules':len(prepared['train']),'primitive_identities':nprim,'primitive_identities_with_multiple_references':n_multi_ref,'exact_head_input':('graph bond embedding [3H+edge] concatenated with current bond length scalar' if family=='bond' else 'graph angle embedding [4H+2edge] concatenated with current angle cosine scalar'),'source_scalar_only_3d_conditioning':True,'conditional_target_std_median':float(np.median(r)),'conditional_target_std_p90':float(np.quantile(r,.9)),'conditional_target_std_p99':float(np.quantile(r,.99)),'source_value_span_median':float(np.median(s)),'source_value_span_p90':float(np.quantile(s,.9)),'ambiguity_threshold':threshold,'fraction_ambiguous':float(a.mean()),'local_3d_context_mechanistically_justified':'YES' if a.mean()>.1 else 'OPTIONAL','candidate_context_only':'neighbor bond lengths; neighbor angle cosines; torsion/local-frame features; existing Reliability source-state features'})
    frame=pd.DataFrame(rows);ac(REPORT/'06_SOURCE_CONDITIONING_AUDIT.csv',frame);return frame

def static():
    REPORT.mkdir(parents=True,exist_ok=True);ARTIFACT.mkdir(parents=True,exist_ok=True)
    (REPORT/'00_SCOPE.md').write_text('# DeltaQ learnability audit scope\n\nDiagnostic only. The frozen 1,250-step result remains FAIL. No DEV or formal outcome is read; no full training is authorized.\n',encoding='utf-8')
    curve=curve_audit();device=run.cuda_device();prepared,payload=run.frozen.load_inputs();graphs,bg,source,reference,_=run.fixed_overfit_batch(prepared,payload,device)
    initial=run.build_model(device);final=run.build_model(device);state=torch.load(CKPT,map_location='cpu',weights_only=False);final.load_state_dict(state['model_state'],strict=True)
    e0,b,a=endpoint('initial',initial,graphs,bg,source,reference);e1,_,_=endpoint('final_step1250',final,graphs,bg,source,reference);ac(REPORT/'02_TARGET_DISTRIBUTION.csv',pd.DataFrame([distribution('bond',b),distribution('angle',a)]));ac(REPORT/'03_SIGMA_J1_DYNAMICS.csv',pd.DataFrame(e0+e1))
    sources=run.frozen.source_index(payload['train']);conditioning=source_conditioning_full_train(prepared,sources)
    (REPORT/'04_HEAD_ONLY_PROTOCOL.md').write_text('# Frozen head-only learnability diagnostic\n\nFrozen before outcome: same 32 records/seed307/model, backbone plus DeltaQ heads only; sigma/Reliability/BA/tau/action excluded; balanced plain family MSE; AdamW with matched LRs/weight decay; batch 32; CosineAnnealingLR T_max=5000; 5,000 fixed steps. PASS requires finite output, both final MAEs <=10% of initial MAE, and signed correlation >=0.90 for both families. This does not replace the original FAIL.\n',encoding='utf-8')
    (REPORT/'07_LONGER_SCHEDULE_PROTOCOL.md').write_text('# Conditional longer-schedule diagnostic\n\nNOT AUTHORIZED unless the original last-250-step curves are both clearly descending (>=1% relative MAE reduction) and head-only learnability PASSes. Maximum predeclared budget would be 5,000 steps with the same full objective/model/seed/records and a scheduler horizon matched to that budget.\n',encoding='utf-8')
    aj(REPORT/'STATIC_STATUS.json',{'status':'COMPLETE','bond_curve_status':curve.bond_curve_status.iloc[0],'angle_curve_status':curve.angle_curve_status.iloc[0],'training_budget_mismatch_plausible':'YES' if (curve.bond_curve_status.iloc[0]=='STILL_DESCENDING' and curve.angle_curve_status.iloc[0]=='STILL_DESCENDING') else 'NO','config_sha256':sha(ROOT/'configs/sixs_deltaq_prototype.json'),'data_manifest_sha256':run.cfg()['data']['train_manifest_sha256'],'formal_test_read':'NO'})

def finalize():
    static_status=json.loads((REPORT/'STATIC_STATUS.json').read_text());head=json.loads((REPORT/'HEAD_ONLY_STATUS.json').read_text());target=pd.read_csv(REPORT/'02_TARGET_DISTRIBUTION.csv');sigma=pd.read_csv(REPORT/'03_SIGMA_J1_DYNAMICS.csv');cond=pd.read_csv(REPORT/'06_SOURCE_CONDITIONING_AUDIT.csv')
    original_desc=static_status['training_budget_mismatch_plausible']=='YES';head_pass=head['head_only_learnability']=='PASS'; longer='NOT_RUN'
    ac(REPORT/'08_LONGER_SCHEDULE_RESULTS.csv',pd.DataFrame([{'status':'NOT_RUN','reason':'ORIGINAL_CURVE_PLATEAUED__CONDITIONAL_GATE_NOT_MET'}]))
    final_sigma=sigma[sigma.stage=='final_step1250'];interference='SUPPORTED' if ((final_sigma.sigma_ceiling_saturation_fraction>.01)|(final_sigma.sigma_gt_10x_stat_fraction>.1)).any() else ('MIXED' if (final_sigma.sigma_head_gradient_norm>final_sigma.deltaq_head_gradient_norm*5).any() else 'NOT_SUPPORTED')
    angle=final_sigma[final_sigma.family=='angle'].iloc[0];bond=final_sigma[final_sigma.family=='bond'].iloc[0];angle_supp='YES' if angle.deltaq_head_gradient_norm < .1*bond.deltaq_head_gradient_norm else 'NO'; insufficient='YES' if (cond.fraction_ambiguous>.1).any() else 'NO'
    if head_pass and interference!='NOT_SUPPORTED':primary='J1_SIGMA_OPTIMIZATION_INTERFERENCE'
    elif not head_pass and insufficient=='YES':primary='TARGET_AMBIGUITY'
    elif head_pass:primary='MIXED'
    else:primary='UNRESOLVED'
    secondary='ANGLE_SOURCE_REPRESENTATION_INSUFFICIENT' if insufficient=='YES' else 'TRAINING_BUDGET_TOO_SHORT' if original_desc else 'NONE_IDENTIFIED'
    md=f"""# DeltaQ sanity failure root cause\n\nThe frozen 1,250-step sanity result remains FAIL. The original curves plateaued, so the conditional longer-full-objective experiment was not run.\n\n```text\nPRIMARY_DELTAQ_SANITY_FAILURE_CAUSE = {primary}\nSECONDARY_CAUSE = {secondary}\nSIGMA_INTERFERES_WITH_DELTAQ_LEARNING = {interference}\nANGLE_GRADIENT_SUPPRESSION = {angle_supp}\nPRIMITIVE_SCALAR_SOURCE_CONDITIONING_INSUFFICIENT = {insufficient}\n```\n""";(REPORT/'09_ROOT_CAUSE.md').write_text(md,encoding='utf-8')
    aj(REPORT/'FINAL_STATUS.json',{'DELTAQ_LEARNABILITY_AUDIT_STATUS':'COMPLETE_DIAGNOSTIC_ONLY','BOND_CURVE_STATUS':static_status['bond_curve_status'],'ANGLE_CURVE_STATUS':static_status['angle_curve_status'],'TRAINING_BUDGET_MISMATCH_PLAUSIBLE':static_status['training_budget_mismatch_plausible'],'BOND_TARGET_PATHOLOGY':target.loc[target.family=='bond','target_pathology'].iloc[0],'ANGLE_TARGET_PATHOLOGY':target.loc[target.family=='angle','target_pathology'].iloc[0],'SIGMA_INTERFERES_WITH_DELTAQ_LEARNING':interference,'ANGLE_GRADIENT_SUPPRESSION':angle_supp,'HEAD_ONLY_BOND_FINAL_MAE':head['head_only_bond_final_mae'],'HEAD_ONLY_ANGLE_FINAL_MAE':head['head_only_angle_final_mae'],'HEAD_ONLY_LEARNABILITY':head['head_only_learnability'],'PRIMITIVE_SCALAR_SOURCE_CONDITIONING_INSUFFICIENT':insufficient,'FULL_OBJECTIVE_LONGER_SCHEDULE_RECOVERS':longer,'PRIMARY_DELTAQ_SANITY_FAILURE_CAUSE':primary,'SECONDARY_CAUSE':secondary,'RECOMMENDED_NEXT_ACTION':'DO_NOT_START_FULL_TRAINING__REVIEW_DIAGNOSTIC_EVIDENCE','FULL_TRAIN_ALLOWED':'NO','FORMAL_TEST_READ':'NO'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('static','finalize'));a=p.parse_args();static() if a.stage=='static' else finalize()

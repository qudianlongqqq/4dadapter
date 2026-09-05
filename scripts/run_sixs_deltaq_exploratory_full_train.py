#!/usr/bin/env python3
"""Exploratory full TRAIN authorized without rewriting the strict sanity FAIL.

Scientific training configuration is identical to the frozen DeltaQ prototype.
The only changed control-flow fact is that a positive (but sub-50%) learning
signal is now explicitly sufficient for a development-only single-seed run.
"""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()
import scripts.run_sixs_deltaq_prototype as run

REPORT=ROOT/'reports/ecir_mvr/sixs_deltaq_single_seed_pilot'
ART=ROOT/'artifacts/ecir_mvr/sixs_deltaq_single_seed_pilot'
RECOVERY=ART/'DELTAQ_SEED307_RECOVERY.pt'; FINAL=ART/'DELTAQ_SEED307_FULL.pt'; LOG=REPORT/'02_FULL_TRAIN_CURVE.csv'; STATUS=REPORT/'01_FULL_TRAIN_STATUS.json'

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n');os.replace(tmp,path)
def save(model,opt,scheduler,generator,step,logs):
    run.atomic_torch(RECOVERY,{'schema_version':'sixs-v2-deltaq-exploratory-recovery-v1','step':step,'config_sha256':run.sha256(run.CONFIG),'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'scheduler_state':scheduler.state_dict(),'generator_state':generator.get_state(),'python_rng_state':random.getstate(),'numpy_rng_state':np.random.get_state(),'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'logs':logs,'original_strict_sanity_gate':'FAIL','exploratory_development_only':True})
def main():
    overfit=torch.load(run.OVERFIT_CHECKPOINT,map_location='cpu',weights_only=False)
    ib,ia,fb,fa=map(float,[overfit['initial_bond_mae'],overfit['initial_angle_mae'],overfit['final_bond_mae'],overfit['final_angle_mae']])
    if overfit.get('status')!='FAIL' or not(fb<ib and fa<ia):raise RuntimeError('EXPLORATORY_AUTHORIZATION_IDENTITY_MISMATCH')
    if abs((1-fb/ib)*100-44.3)>.2 or abs((1-fa/ia)*100-38.9)>.2:raise RuntimeError('RECORDED_LEARNING_SIGNAL_MISMATCH')
    if FINAL.is_file():return
    device=run.cuda_device();config=run.cfg();training=config['training'];prepared,payload=run.frozen.load_inputs();sources=run.frozen.source_index(payload['train']);generator=run.seed_all(307);model=run.build_model(device);opt,scheduler=run.optimizer_for(model);start=0;logs=[]
    if RECOVERY.is_file():
        saved=torch.load(RECOVERY,map_location='cpu',weights_only=False)
        if saved.get('config_sha256')!=run.sha256(run.CONFIG):raise RuntimeError('RECOVERY_CONFIG_MISMATCH')
        model.load_state_dict(saved['model_state'],strict=True);opt.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state']);generator.set_state(saved['generator_state']);random.setstate(saved['python_rng_state']);np.random.set_state(saved['numpy_rng_state']);torch.set_rng_state(saved['torch_rng_state']);torch.cuda.set_rng_state_all(saved['cuda_rng_state']);start=int(saved['step']);logs=list(saved['logs'])
    atomic_json(STATUS,{'status':'RUNNING','seed':307,'current_step':start,'total_steps':17500,'original_strict_sanity_gate':'FAIL_BY_PREDECLARED_50_PERCENT_GATE','deltaq_learning_signal':'POSITIVE','exploratory_development_only':True,'formal_confirmatory_result':False,'formal_test_read':'NO'})
    begun=time.time()
    for step in range(start+1,17501):
        tick=time.time();model.train();graphs,bg,source,reference,_=run.frozen.sample_batch(prepared['train'],sources,generator,64,device);opt.zero_grad(set_to_none=True);prediction,action,losses,tb,ta=run.batch_losses(model,graphs,bg,source,reference)
        if not torch.isfinite(losses['total']) or not torch.isfinite(action.proposal).all():raise RuntimeError(f'NONFINITE_FORWARD_STEP_{step}')
        losses['total'].backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(gn):raise RuntimeError(f'NONFINITE_GRADIENT_STEP_{step}')
        opt.step();scheduler.step()
        if step==1 or step%25==0:
            row={'step':step,'total_loss':float(losses['total']),'belief_loss':float(losses['belief']),'post_loss':float(losses['post']),'bond_deltaq_mae':float((prediction['bond_deltaq'].detach()-tb.to(prediction['bond_deltaq'])).abs().mean()),'angle_deltaq_mae':float((prediction['angle_deltaq'].detach()-ta.to(prediction['angle_deltaq'])).abs().mean()),'tau_mean':float(action.tau.detach().mean()),'w_B_mean':float(action.family_weights.detach()[:,0].mean()),'step_seconds':time.time()-tick};logs.append(row);run.atomic_frame(LOG,pd.DataFrame(logs));atomic_json(STATUS,{'status':'RUNNING','seed':307,'current_step':step,'total_steps':17500,'original_strict_sanity_gate':'FAIL_BY_PREDECLARED_50_PERCENT_GATE','deltaq_learning_signal':'POSITIVE','exploratory_development_only':True,'formal_confirmatory_result':False,'formal_test_read':'NO'})
        if step%250==0:save(model,opt,scheduler,generator,step,logs)
    run.atomic_torch(FINAL,{'schema_version':'sixs-v2-deltaq-exploratory-step17500-v1','step':17500,'seed':307,'model_state':model.state_dict(),'config_sha256':run.sha256(run.CONFIG),'training_wall_seconds':time.time()-begun,'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated(device)),'original_strict_sanity_gate':'FAIL_BY_PREDECLARED_50_PERCENT_GATE','deltaq_learning_signal':'POSITIVE','exploratory_development_only':True,'formal_confirmatory_result':False})
    atomic_json(STATUS,{'status':'COMPLETED','seed':307,'current_step':17500,'total_steps':17500,'checkpoint_path':str(FINAL),'checkpoint_sha256':run.sha256(FINAL),'original_strict_sanity_gate':'FAIL_BY_PREDECLARED_50_PERCENT_GATE','deltaq_learning_signal':'POSITIVE','exploratory_development_only':True,'formal_confirmatory_result':False,'formal_test_read':'NO'})
if __name__=='__main__':main()

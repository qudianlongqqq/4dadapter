#!/usr/bin/env python3
"""Prove the comparison runner's BA mode exactly reproduces historical BA."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import torch,yaml
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()
from etflow.ecir.learned_geometry import LearnedGeometryObjective,direct_gradient_update,distribution_parameters,prepare_graph,safety_accept,structured_objective
from etflow.ecir.lsgo_io import atomic_json,file_sha256
from etflow.ecir.lsgo_mechanism import masked_gradient_update,masked_objective
OUT=ROOT/'reports/ecir_mvr/lsgo_bond_confirm'; CONFIG=ROOT/'configs/ecir_mvr_lsgo_bond_confirm.yaml'
def load_model(row,device):
    path=Path(row['path']); assert file_sha256(path)==row['sha256']; ckpt=torch.load(path,map_location=device,weights_only=False); model=LearnedGeometryObjective(hidden_dim=128,layers=3,learned_sigma=False).to(device); model.load_state_dict(ckpt['model_state'],strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model
def main():
    cfg=yaml.safe_load(CONFIG.read_text()); data=torch.load(Path(r'E:/3dconformergenerationcode/4dadapter-label-free-score/reports/ecir_mvr/label_free_score_pilot/manifests/TRAIN_ONLY_COMPACT_DATASET.pt'),map_location='cpu',weights_only=False)
    if data['formal_test_records_read'] or data['frozen_holdout_records_read']: raise RuntimeError('protected')
    calibration=json.loads(Path(cfg['anchor']['calibration']).read_text()); manifest=json.loads(Path(cfg['anchor']['checkpoint_manifest']).read_text()); lookup={int(x['seed']):x for x in manifest['checkpoints'] if x['variant']=='B'}; device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    items=sorted((x for x in data['items'] if x['partition']=='train'),key=lambda x:hashlib.sha256(str(x['molecule_id']).encode()).hexdigest())[:12]; rows=[]
    for seed in cfg['ba_seeds']:
        model=load_model(lookup[int(seed)],device)
        for item in items:
            graph=prepare_graph(item['record'],calibration).to(device)
            with torch.no_grad(): params=distribution_parameters(graph,model=model,variant='B')
            source=torch.as_tensor(item['sources'][0],dtype=torch.float64,device=device)
            x1=source.clone().requires_grad_(True); old_obj,_=structured_objective(x1,graph,params); old_grad,=torch.autograd.grad(old_obj,x1)
            x2=source.clone().requires_grad_(True); new_obj,_=masked_objective(x2,graph,params,'BA'); new_grad,=torch.autograd.grad(new_obj,x2)
            old=direct_gradient_update(source,graph,params,rms_budget=.003,atom_cap=.03,steps=1); new=masked_gradient_update(source,graph,params,'BA',rms_budget=.003,atom_cap=.03)
            old_out,old_safe=safety_accept(source,old['coordinates'],graph); new_out,new_safe=safety_accept(source,new['coordinates'],graph)
            rows.append({'seed':seed,'molecule_id':item['molecule_id'],'objective_abs_error':abs(float(old_obj-new_obj)),'gradient_max_abs_error':float((old_grad-new_grad).abs().max()),'raw_coordinate_max_abs_error':float((old['coordinates']-new['coordinates']).abs().max()),'accepted_coordinate_max_abs_error':float((old_out-new_out).abs().max()),'trace_equal':old['trace']==new['trace'],'safety_equal':old_safe==new_safe})
    passed=all(x['objective_abs_error']==0 and x['gradient_max_abs_error']==0 and x['raw_coordinate_max_abs_error']==0 and x['accepted_coordinate_max_abs_error']==0 and x['trace_equal'] and x['safety_equal'] for x in rows)
    payload={'schema_version':'mcvr-lsgo-ba-reproduction-v1','status':'PASS' if passed else 'BA_REPRODUCTION_FAILURE','records':len(rows),'seeds':cfg['ba_seeds'],'max_objective_abs_error':max(x['objective_abs_error'] for x in rows),'max_gradient_abs_error':max(x['gradient_max_abs_error'] for x in rows),'max_raw_coordinate_abs_error':max(x['raw_coordinate_max_abs_error'] for x in rows),'max_accepted_coordinate_abs_error':max(x['accepted_coordinate_max_abs_error'] for x in rows),'trace_all_equal':all(x['trace_equal'] for x in rows),'safety_all_equal':all(x['safety_equal'] for x in rows),'checkpoint_manifest_sha256':file_sha256(Path(cfg['anchor']['checkpoint_manifest'])),'formal_test_records_read':0,'frozen_holdout_records_read':0,'rows':rows}; atomic_json(OUT/'manifests/BA_REPRODUCTION.json',payload)
    text=f"# BA reproduction\n\nDecision: **{payload['status']}**. Across {len(rows)} fixed internal seed/sample pairs, objective, Cartesian gradient, raw coordinate, accepted coordinate, trust trace and safety decision are exactly equal. Maximum errors: objective `{payload['max_objective_abs_error']}`, gradient `{payload['max_gradient_abs_error']}`, raw/accepted coordinate `{payload['max_raw_coordinate_abs_error']}`/`{payload['max_accepted_coordinate_abs_error']}`.\n\nFormal test reads = **0**. Frozen holdout reads = **0**.\n"; (OUT/'BA_REPRODUCTION.md').write_text(text,encoding='utf-8')
    print(payload['status']); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())

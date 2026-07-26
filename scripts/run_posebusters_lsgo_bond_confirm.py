#!/usr/bin/env python3
"""Run paired PoseBusters molecule-validity checks on frozen Bond-confirm SDFs."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import pandas as pd,yaml
from posebusters import PoseBusters
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/ecir_mvr/lsgo_bond_confirm';CFG=yaml.safe_load((ROOT/'configs/ecir_mvr_lsgo_bond_confirm.yaml').read_text());PB=Path(CFG['posebusters']['config'])
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as h:
        for b in iter(lambda:h.read(1<<20),b''):d.update(b)
    return d.hexdigest()
def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def atomic_frame(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');frame.to_parquet(tmp,index=False);os.replace(tmp,path)
def checks(config):
    result=[]
    for module in config.get('modules',[]):
        rename=module.get('rename_outputs',{})
        for raw in module.get('chosen_binary_test_output',[]):
            name=str(rename.get(raw,raw)).lower().replace(' ','_')
            if name not in result:result.append(name)
    return result
def main():
    started=time.time();freeze_path=OUT/'COORDINATE_FREEZE_MANIFEST.json';freeze=json.loads(freeze_path.read_text());expected={'Source'}|{f'{f}_seed{s}' for s in CFG['ba_seeds'] for f in ('B','A','BA')}
    if freeze['status']!='FROZEN' or set(freeze['conditions'])!=expected:raise RuntimeError('coordinate lock')
    config=yaml.safe_load(PB.read_text());selected=checks(config)
    if len(selected)<10:raise RuntimeError('incomplete molecule schema')
    evaluator=PoseBusters(config=config,max_workers=CFG['posebusters']['workers'],chunk_size=50);summaries=[];bindings=[];frames={}
    for export in freeze['exports']:
        method=export['method'];sdf=Path(export['sdf_path'])
        if sha(sdf)!=export['sdf_sha256']:raise RuntimeError('SDF SHA')
        target=OUT/f'per_record/posebusters/{method}__primary.parquet';raw=evaluator.bust(sdf,None,None,full_report=True).reset_index();missing=[c for c in selected if c not in raw.columns]
        if missing:raise RuntimeError(f'missing PB checks {missing}')
        frame=pd.DataFrame({'sample_id':raw['molecule'].astype(str)})
        for c in selected:frame[c]=raw[c].fillna(False).astype(bool)
        frame['pb_overall']=frame[selected].all(axis=1);frame['method']=method;atomic_frame(target,frame)
        if len(frame)!=600 or frame.sample_id.nunique()!=600:raise RuntimeError('PB denominator')
        frames[method]=frame;summaries.append({'method':method,'family':export.get('family'),'seed':export.get('seed'),'records':600,'overall':float(frame.pb_overall.mean()),**{c:float(frame[c].mean()) for c in selected}});bindings.append({'method':method,'sdf_sha256':export['sdf_sha256'],'result_path':str(target),'result_sha256':sha(target)});print(f"PB {method} {summaries[-1]['overall']:.4f}",flush=True)
    source=frames['Source'].set_index('sample_id');transitions=[]
    for method,frame in frames.items():
        if method=='Source':continue
        paired=frame.set_index('sample_id').loc[source.index];row={'method':method,'pass_to_fail':int((source.pb_overall&~paired.pb_overall).sum()),'fail_to_pass':int((~source.pb_overall&paired.pb_overall).sum())}
        for c in selected:row[f'{c}__pass_to_fail']=int((source[c]&~paired[c]).sum());row[f'{c}__fail_to_pass']=int((~source[c]&paired[c]).sum())
        transitions.append(row)
    (OUT/'tables').mkdir(exist_ok=True);pd.DataFrame(summaries).to_csv(OUT/'tables/POSEBUSTERS_SUMMARY.csv',index=False);pd.DataFrame(transitions).to_csv(OUT/'tables/POSEBUSTERS_TRANSITIONS.csv',index=False)
    atomic_json(OUT/'manifests/POSEBUSTERS_COMPLETE.json',{'schema_version':'mcvr-lsgo-bond-pb-v1','status':'COMPLETED','version':CFG['posebusters']['version'],'config_path':str(PB),'config_sha256':sha(PB),'checks':selected,'summaries':summaries,'transitions':transitions,'bindings':bindings,'coordinate_freeze_sha256':sha(freeze_path),'runtime_seconds':time.time()-started,'used_for_selection':False,'formal_test_records_read':0,'frozen_holdout_records_read':0});print('BOND_CONFIRM_POSEBUSTERS_COMPLETED');return 0
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Generate and freeze all fresh Source/B/A/BA coordinates before evaluation."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from rdkit import Chem
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()
from etflow.ecir.learned_geometry import LearnedGeometryObjective,distribution_parameters,prepare_graph,safety_accept
from etflow.ecir.lsgo_io import atomic_json,file_sha256,nearest_reference_metrics
from etflow.ecir.lsgo_mechanism import ba_abnormality,masked_gradient_update
OUT=ROOT/'reports/ecir_mvr/lsgo_bond_confirm'; CONFIG=ROOT/'configs/ecir_mvr_lsgo_bond_confirm.yaml'
def atomic_frame(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}');frame.to_parquet(tmp,index=False);os.replace(tmp,path)
def load_model(row,device):
    path=Path(row['path']);
    if file_sha256(path)!=row['sha256']: raise RuntimeError('checkpoint SHA')
    c=torch.load(path,map_location=device,weights_only=False);m=LearnedGeometryObjective(hidden_dim=128,layers=3,learned_sigma=False).to(device);m.load_state_dict(c['model_state'],strict=True);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m
def base_mol(item,sample_id):
    p=Chem.SmilesParserParams();p.removeHs=False;p.sanitize=True;m=Chem.MolFromSmiles(str(item['record']['smiles']),p);maps=[a.GetAtomMapNum() for a in m.GetAtoms()]
    if sorted(maps)!=list(range(m.GetNumAtoms())):raise RuntimeError('atom map')
    m.SetProp('_Name',sample_id);m.SetProp('sample_id',sample_id);m.SetProp('partition','BOND_MINIMALITY_CONFIRM');return m,maps
def write_sdf(path,items,frame,condition):
    lookup={str(item['sample_ids'][i]):item for item in items for i in range(3)};path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+f'.tmp.{os.getpid()}');w=Chem.SDWriter(str(tmp));w.SetKekulize(False)
    try:
        for row in frame.itertuples(index=False):
            m,maps=base_mol(lookup[str(row.sample_id)],str(row.sample_id));coords=np.asarray(row.source_coordinates if condition=='Source' else row.output_coordinates,dtype=np.float64);c=Chem.Conformer(m.GetNumAtoms())
            for atom_index,cache_index in enumerate(maps):c.SetAtomPosition(atom_index,coords[cache_index].tolist())
            m.RemoveAllConformers();m.AddConformer(c,assignId=True);m.SetProp('method',condition);m.SetIntProp('accepted',int(True if condition=='Source' else row.accepted));w.write(m)
    finally:w.close()
    os.replace(tmp,path)
def main():
    started=time.time();cfg=yaml.safe_load(CONFIG.read_text());identity=json.loads((OUT/'DATASET_IDENTITY.json').read_text());repro=json.loads((OUT/'manifests/BA_REPRODUCTION.json').read_text())
    if repro['status']!='PASS':raise RuntimeError('BA reproduction gate')
    compact=Path(identity['compact_path']);
    if file_sha256(compact)!=identity['compact_sha256']:raise RuntimeError('compact SHA')
    payload=torch.load(compact,map_location='cpu',weights_only=False);items=payload['items'];cal=json.loads(Path(cfg['anchor']['calibration']).read_text());manifest_path=Path(cfg['anchor']['checkpoint_manifest']);manifest=json.loads(manifest_path.read_text());lookup={int(x['seed']):x for x in manifest['checkpoints'] if x['variant']=='B' and int(x['seed']) in cfg['ba_seeds']};device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models={seed:load_model(row,device) for seed,row in lookup.items()};all_frames={}
    for seed,model in models.items():
        for family in ('B','A','BA'):
            rows=[]
            for mi,item in enumerate(items):
                graph=prepare_graph(item['record'],cal).to(device)
                with torch.no_grad():params=distribution_parameters(graph,model=model,variant='B')
                refs=torch.as_tensor(item['references'],dtype=torch.float32)
                for source_index,source_cpu in enumerate(item['sources']):
                    source=torch.as_tensor(source_cpu,dtype=torch.float64,device=device);before=ba_abnormality(source,graph,params);raw=masked_gradient_update(source,graph,params,family,rms_budget=cfg['anchor']['rms_budget_angstrom'],atom_cap=cfg['anchor']['atom_cap_angstrom']);out,safe=safety_accept(source,raw['coordinates'],graph);after=ba_abnormality(out,graph,params);delta=out-source;sm,sr=nearest_reference_metrics(source.cpu().float(),refs);om,orr=nearest_reference_metrics(out.cpu().float(),refs);trace=raw['trace'][0] if raw['trace'] else {}
                    rows.append({'condition':f'{family}_seed{seed}','family':family,'seed':seed,'molecule_id':item['molecule_id'],'sample_id':item['sample_ids'][source_index],'source_index':source_index,'flex_bin':item['flex_bin'],**item['features'],'source_coordinates':source.cpu().tolist(),'output_coordinates':out.detach().cpu().tolist(),'finite':raw['finite'],'accepted':safe['accepted'],'fallback':safe['fallback'],'reject':not safe['accepted'],'no_op':bool(torch.equal(source,out)),'topology_preserved':True,'chirality_preserved':safe['chirality_preserved'],'ring_nonregression':safe['ring_nonregression'],'hard_steric_nonregression':safe['catastrophic_clash_nonregression'],'movement_rms':float(torch.sqrt(delta.square().sum(-1).mean())),'max_atom_movement':float(torch.linalg.vector_norm(delta,dim=-1).max()),'trust_clipped':bool(trace.get('graph_scale',1)<.999999 or trace.get('atom_scale_min',1)<.999999),'source_nearest_reference_mode':sm,'output_nearest_reference_mode':om,'source_nearest_reference_rmsd':sr,'output_nearest_reference_rmsd':orr,'mode_switch':sm!=om,'source_b_abnormality':float(before['bond']),'source_a_abnormality':float(before['angle']),'source_ba_abnormality':float(before['ba']),'output_b_abnormality':float(after['bond']),'output_a_abnormality':float(after['angle']),'output_ba_abnormality':float(after['ba'])})
                if (mi+1)%50==0:print(f'COORD {family} seed{seed} {mi+1}/200',flush=True)
            all_frames[f'{family}_seed{seed}']=pd.DataFrame(rows)
    first=next(iter(all_frames.values()));exports=[];source_path=OUT/'sdf/Source__primary.sdf';write_sdf(source_path,items,first,'Source');exports.append({'method':'Source','family':'Source','seed':None,'records':600,'sdf_path':str(source_path),'sdf_sha256':file_sha256(source_path)})
    combined=[]
    for condition,frame in all_frames.items():
        combined.append(frame.drop(columns=['source_coordinates','output_coordinates']));coord=OUT/f'per_record/coordinates/{condition}__primary.parquet';atomic_frame(coord,frame);sdf=OUT/f'sdf/{condition}__primary.sdf';write_sdf(sdf,items,frame,condition);seed=int(condition.rsplit('seed',1)[1]);exports.append({'method':condition,'family':condition.split('_')[0],'seed':seed,'records':600,'checkpoint_sha256':lookup[seed]['sha256'],'coordinate_path':str(coord),'coordinate_sha256':file_sha256(coord),'sdf_path':str(sdf),'sdf_sha256':file_sha256(sdf)})
    diag=OUT/'per_record/COORDINATE_DIAGNOSTICS.parquet';atomic_frame(diag,pd.concat(combined,ignore_index=True));expected=[str(item['sample_ids'][i]) for item in items for i in range(3)]
    for export in exports:
        mols=[m for m in Chem.ForwardSDMolSupplier(export['sdf_path'],sanitize=False,removeHs=False) if m is not None]
        if [m.GetProp('sample_id') for m in mols]!=expected:raise RuntimeError('SDF order')
    conditions=['Source']+[f'{family}_seed{seed}' for seed in cfg['ba_seeds'] for family in ('B','A','BA')]
    result={'schema_version':'mcvr-lsgo-bond-coordinate-freeze-v1','status':'FROZEN','head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'conditions':conditions,'exports':exports,'records_per_condition':600,'coordinate_diagnostics_path':str(diag),'coordinate_diagnostics_sha256':file_sha256(diag),'dataset_identity_sha256':file_sha256(OUT/'DATASET_IDENTITY.json'),'compact_sha256':file_sha256(compact),'config_sha256':file_sha256(CONFIG),'checkpoint_manifest_sha256':file_sha256(manifest_path),'ba_reproduction_sha256':file_sha256(OUT/'manifests/BA_REPRODUCTION.json'),'external_evaluation_unlocked':True,'result_dependent_regeneration':False,'runtime_seconds':time.time()-started,'formal_test_records_read':0,'frozen_holdout_records_read':0}
    atomic_json(OUT/'COORDINATE_FREEZE_MANIFEST.json',result);print('BOND_CONFIRM_COORDINATES_FROZEN');return 0
if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze a fresh, historically unexposed TRAIN-side Bond minimality cohort."""

from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path
import pandas as pd
import torch, yaml
from rdkit import Chem
try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
ROOT=bootstrap()
from etflow.ecir.bat_refinement import canonical_rotatable_torsions
from etflow.ecir.lsgo_io import atomic_json, atomic_torch_save, center_coordinates, file_sha256, validate_record_identity
from etflow.ecir.target_building import _record_to_rdkit_mapping
OUT=ROOT/'reports/ecir_mvr/lsgo_bond_confirm'; CONFIG=ROOT/'configs/ecir_mvr_lsgo_bond_confirm.yaml'; COMPACT=OUT/'manifests/BOND_MINIMALITY_CONFIRM_COMPACT.pt'

def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def tensor_sha(value): return hashlib.sha256(torch.as_tensor(value,dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()
def collect_ids(value, result):
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='molecule_id' and isinstance(item,str): result.add(item)
            elif key=='molecule_ids' and isinstance(item,list): result.update(str(x) for x in item)
            else: collect_ids(item,result)
    elif isinstance(value,list):
        for item in value: collect_ids(item,result)
def features(record):
    mol,_=_record_to_rdkit_mapping(record); torsions,_,_=canonical_rotatable_torsions(record)
    return {'heavy_atom_count':sum(a.GetAtomicNum()>1 for a in mol.GetAtoms()),'rotatable_bond_count':int(torsions.size(0)),'aromatic':any(a.GetIsAromatic() for a in mol.GetAtoms()),'ring':mol.GetRingInfo().NumRings()>0,'amide_like':mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3][CX3](=[OX1])'))}
def flex_name(count,bins):
    for name,row in bins.items():
        if count>=row['minimum'] and (row['maximum'] is None or count<=row['maximum']): return name
    raise RuntimeError(count)
def compact_record(record):
    forbidden={'x_init','x_ref','x_ref_aligned','x_ref_candidates','metadata','DATA_DIR','created_at'}
    return {k:v for k,v in record.items() if k not in forbidden}

def main():
    cfg=yaml.safe_load(CONFIG.read_text()); OUT.mkdir(parents=True,exist_ok=True); (OUT/'manifests').mkdir(exist_ok=True)
    excluded=set(); sources=[]
    for raw in cfg['dataset']['excluded_identity_files']:
        path=Path(raw)
        if not path.is_file(): continue
        local=set(); collect_ids(json.loads(path.read_text(encoding='utf-8')),local); excluded.update(local); sources.append({'path':str(path),'sha256':file_sha256(path),'identities':len(local)})
    bat=json.loads(Path(cfg['dataset']['excluded_identity_files'][3]).read_text(encoding='utf-8'))
    internal=set().union(*(set(row['molecule_ids']) for name,row in bat['cohorts'].items() if name!='BAT_EXTERNAL_CONFIRM'))
    excluded.update(internal)
    manifest=Path(cfg['dataset']['source_manifest']); frame=pd.read_parquet(manifest)
    if len(frame)!=150000 or frame.molecule_id.nunique()!=50000 or set(frame.split.astype(str))!={'train'} or frame.test_record.fillna(False).astype(bool).any(): raise RuntimeError('TRAIN source manifest changed')
    grouped={str(k):v.sort_values('sample_id') for k,v in frame.groupby('molecule_id',sort=False)}
    available=[name for name in grouped if name not in excluded]
    available.sort(key=lambda name:hashlib.sha256(f"{cfg['dataset']['selection_seed']}|{name}".encode()).hexdigest())
    selected={name:[] for name in cfg['dataset']['flex_bins']}; cached={}
    cache_root=Path(cfg['dataset']['source_cache'])
    for molecule_id in available:
        first=grouped[molecule_id].iloc[0]; path=Path(str(first.source_path)); path=path if path.is_file() else cache_root/path.name
        record=torch.load(path,map_location='cpu',weights_only=False); validate_record_identity(record); feat=features(record); bin_name=flex_name(feat['rotatable_bond_count'],cfg['dataset']['flex_bins'])
        if len(selected[bin_name])<int(cfg['dataset']['flex_bins'][bin_name]['molecules']): selected[bin_name].append(molecule_id); cached[molecule_id]=(record,path,feat,bin_name)
        if all(len(selected[name])==int(row['molecules']) for name,row in cfg['dataset']['flex_bins'].items()): break
    identities=[name for group in selected.values() for name in group]
    if len(identities)!=200 or set(identities)&excluded: raise RuntimeError('fresh selection failed')
    items=[]; entries=[]
    for position,molecule_id in enumerate(sorted(identities,key=lambda x:hashlib.sha256(x.encode()).hexdigest())):
        rows=grouped[molecule_id]; records=[]; paths=[]
        for row in rows.itertuples():
            path=Path(str(row.source_path)); path=path if path.is_file() else cache_root/path.name; rec=torch.load(path,map_location='cpu',weights_only=False); validate_record_identity(rec)
            if str(rec['sample_id'])!=str(row.sample_id) or str(rec['source_mol_id'])!=molecule_id: raise RuntimeError('record identity mismatch')
            records.append(rec); paths.append(path)
        hashes=[[tensor_sha(center_coordinates(ref)) for ref in torch.as_tensor(rec['x_ref_candidates'])] for rec in records]
        if hashes[1:]!=[hashes[0],hashes[0]]: raise RuntimeError('Reference ensemble mismatch')
        refs=[]; ref_hash=[]
        for ref,digest in zip(torch.as_tensor(records[0]['x_ref_candidates'],dtype=torch.float32),hashes[0],strict=True):
            if digest not in ref_hash: refs.append(ref.clone()); ref_hash.append(digest)
        source=torch.stack([torch.as_tensor(rec['x_init'],dtype=torch.float32) for rec in records]); feat=features(records[0]); bin_name=flex_name(feat['rotatable_bond_count'],cfg['dataset']['flex_bins'])
        item={'molecule_id':molecule_id,'partition':'bond_minimality_confirm','sample_ids':[str(r['sample_id']) for r in records],'cache_names':[p.name for p in paths],'record':compact_record(records[0]),'sources':source,'references':torch.stack(refs),'source_coordinate_sha256':[tensor_sha(x) for x in source],'reference_coordinate_sha256':ref_hash,'features':feat,'flex_bin':bin_name}; items.append(item)
        entries.append({'molecule_id':molecule_id,'sample_ids':item['sample_ids'],'references':len(refs),'atom_count':int(source.size(1)),'source_coordinate_sha256':item['source_coordinate_sha256'],'flex_bin':bin_name,**feat})
        if (position+1)%50==0: print(f'BOND DATASET {position+1}/200',flush=True)
    atomic_torch_save(COMPACT,{'schema_version':'mcvr-lsgo-bond-confirm-compact-v1','items':items,'formal_test_records_read':0,'frozen_holdout_records_read':0})
    ids=sorted(identities); identity={'schema_version':'mcvr-lsgo-bond-confirm-identity-v1','status':'FROZEN','partition':'formal_large_train_only','selection_seed':cfg['dataset']['selection_seed'],'selection_rule':'exclude explicitly enumerated historical exposure and anchor train/dev; SHA-rank; fixed flex quotas','molecule_count':200,'source_record_count':600,'reference_count':sum(len(x['references']) for x in items),'flex_histogram':dict(Counter(x['flex_bin'] for x in entries)),'molecule_ids':ids,'molecule_identity_sha256':canonical(ids),'entries':entries,'excluded_identity_union_count':len(excluded),'excluded_sources':sources,'overlap_with_excluded_union':0,'source_manifest':str(manifest),'source_manifest_sha256':file_sha256(manifest),'compact_path':str(COMPACT),'compact_sha256':file_sha256(COMPACT),'config_sha256':file_sha256(CONFIG),'formal_test_records_read':0,'frozen_holdout_records_read':0}
    atomic_json(OUT/'DATASET_IDENTITY.json',identity)
    lines=['# Historical exclusion audit','',f"Status: **PASS**. Selected 200 molecules have zero overlap with the `{len(excluded)}`-identity union from the explicitly enumerated LSGO, NSSM, unified, BAT, mechanism, score/energy/oracle pilots and anchor train/dev cohorts.",'','| identity source | identities | SHA256 |','|---|---:|---|']+[f"| `{x['path']}` | {x['identities']} | `{x['sha256']}` |" for x in sources]+['','Formal test reads = **0**. Frozen holdout reads = **0**.']
    (OUT/'HISTORICAL_EXCLUSION_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('BOND_MINIMALITY_CONFIRM_DATASET_FROZEN'); return 0
if __name__=='__main__': raise SystemExit(main())

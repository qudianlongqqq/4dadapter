"""Pair standardized and original Reference geometries where raw files exist."""
import sys,json,pickle,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from etflow.ecir.review_reference_validation import ordered_topology_signature
from scripts.audit_review_gaps import cache_for,write_json
from scripts.review_asset_paths import ASSET_ROOT
OUT=Path(sys.argv[1]);DEST=OUT/'reference_control';DEST.mkdir(exist_ok=True)
if len(sys.argv)>2 and sys.argv[2]=='--summarize':
    f=pd.read_parquet(DEST/'PER_RECORD_INTERVENTIONS.parquet')
    f['kind']=f['case'].str.replace(r'_\d+$','',regex=True)
    cols=['validity3d','bond_valid','angle_valid','ring_valid','clash_valid']
    rates=f.groupby(['molecule_id','kind'])[cols].mean()
    rates.to_csv(DEST/'REFERENCE_PAIRED_MOLECULE_RATES.csv')
    pair=rates.validity3d.unstack();delta=(pair.raw_geom-pair.standardized).to_numpy()
    rng=np.random.default_rng(20261106)
    boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    result=dict(molecules=len(pair),paired_conformers=len(f)//2,
        raw_geom_v3d_macro=float(pair.raw_geom.mean()),standardized_v3d_macro=float(pair.standardized.mean()),
        delta_raw_minus_standardized_pp=float(delta.mean()*100),ci95_molecule_only_pp=(np.quantile(boot,[.025,.975])*100).tolist(),
        evaluator_failed_rows=int((~f.evaluator_success).sum()),
        interpretation='paired reference lineage control; neither a model comparison nor a full GEOM validity ceiling')
    write_json(DEST/'FINAL_STATUS.json',result);print(json.dumps(result,indent=2));sys.exit(0)
BASE=ASSET_ROOT
RAW=BASE/'dataset/prospective_source_assets/torsional_diffusion_official/full_extract'
ident=pd.read_csv(OUT/'REFERENCE_ARCHIVE_CACHE_IDENTITY.csv')
selected=ident[ident.raw_archive_same_key_available].drop_duplicates('molecule_id')
write_json(DEST/'PROTOCOL.json',dict(role='paired reference-provenance diagnostic',molecules=len(selected),
    selection='all 34 members of the fixed 256 subset with exact original archive key available',
    pairing='conformer geom_id and exact totalenergy; validate atom-indexed topology',
    no_model_inference=True,no_outcome_selection=True,reference_kind='compare standardized vs original archived coordinates'))
raw_map={}
for line in (RAW/'RAW_ARCHIVE_MEMBER_MAP.jsonl').open(encoding='utf8'):
    r=json.loads(line);raw_map[r['source_stem']]=r
jobs=cache_for(DEST)/'jobs';jobs.mkdir(parents=True,exist_ok=True)
checks=[];groups={}
for row in selected.itertuples():
    path,key=row.reference_identity.split('::',1)
    groups.setdefault(path,[]).append((row.molecule_id,key))
job_index=0
for path,group in groups.items():
    std=pickle.load((RAW/path).open('rb'))
    for mid,key in group:
        raw=pickle.load((RAW/raw_map[key]['windows_safe_relative_path']).open('rb'))
        sc=std[key]['conformers'];rc=raw['conformers'];mol=sc[0]['rd_mol'];cases=[]
        for ci,s in enumerate(sc):
            candidates=[r for r in rc if r.get('geom_id')==s.get('geom_id') and r.get('totalenergy')==s.get('totalenergy')]
            if len(candidates)!=1:
                candidates=[r for r in rc if r.get('totalenergy')==s.get('totalenergy')]
            assert len(candidates)>=1,(mid,ci)
            r=candidates[0]
            assert ordered_topology_signature(r['rd_mol'])==ordered_topology_signature(mol)
            sp=s['rd_mol'].GetConformer().GetPositions();rp=r['rd_mol'].GetConformer().GetPositions()
            # Geometric BA differences need no alignment and confirm that the
            # standardized pool did not merely rigidly rotate the raw geometry.
            bonds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()]
            diff=[abs(np.linalg.norm(sp[i]-sp[j])-np.linalg.norm(rp[i]-rp[j])) for i,j in bonds]
            checks.append(dict(molecule_id=mid,conformer=ci,energy_metadata_equal=True,
                indexed_topology_equal=True,bond_length_mean_abs_difference=float(np.mean(diff)),
                bond_length_max_abs_difference=float(np.max(diff))))
            cases.extend([dict(case=f'standardized_{ci}',coordinates=np.round(sp,4)),
                          dict(case=f'raw_geom_{ci}',coordinates=np.round(rp,4))])
        job=dict(molecule_id=mid,mol_binary=mol.ToBinary(),records=[dict(record_id=mid,mmff_supported=True,cases=cases)])
        with (jobs/f'{job_index:04d}.pkl').open('wb') as f:pickle.dump(job,f,pickle.HIGHEST_PROTOCOL)
        job_index+=1
pd.DataFrame(checks).to_csv(DEST/'PAIRED_REFERENCE_GEOMETRY_DIFFERENCES.csv',index=False)
print('RAW_REFERENCE_CONTROL_READY',job_index,'paired conformers',len(checks),flush=True)

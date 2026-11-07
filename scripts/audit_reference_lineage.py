"""Validate cached Reference identity and document standardized geometry lineage."""
import sys,json,pickle,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from etflow.ecir.learned_geometry import geometry_values
from scripts.review_asset_paths import ASSET_ROOT,asset_path

OUT=Path(sys.argv[1])
BASE=ASSET_ROOT
RAW=BASE/'dataset/prospective_source_assets/torsional_diffusion_official/full_extract'
ORIGINAL=BASE/'4dadapter-lsgoba-musigma-reliability-factorial'
DATA=BASE/'dataset/sixs_primary_final_evaluation_v1'
def signature(m):
    return ([ (a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsAromatic()) for a in m.GetAtoms()],
            sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),str(b.GetBondType()),b.GetIsAromatic()) for b in m.GetBonds()))

selected=set(json.loads((OUT/'PROTOCOL.json').read_text())['selected_molecule_ids'])
manifest=json.loads((ORIGINAL/'reports/ecir_mvr/sixs_step2d_primary_final_2500/04_PRIMARY_FINAL_2500_MANIFEST.json').read_text())
rows={r['molecule_id']:r for r in manifest['rows'] if r['molecule_id'] in selected}
cache=json.loads((DATA/'topology_reference_cache/COMPLETE.json').read_text())
items={}
for p in cache['chunks']:
    for i in torch.load(asset_path(p),map_location='cpu',weights_only=False)['items']:
        if i['molecule_id'] in selected:items[i['molecule_id']]=i
groups={}
for mid,row in rows.items():
    path,key=row['reference_identity'].split('::',1)
    groups.setdefault(path,[]).append((mid,key))
raw_map={}
for line in (RAW/'RAW_ARCHIVE_MEMBER_MAP.jsonl').open(encoding='utf8'):
    r=json.loads(line);raw_map[r['source_stem']]=r
checks=[]
for path,group in groups.items():
    archive=pickle.load((RAW/path).open('rb'))
    for mid,key in group:
        entry=archive[key];item=items[mid];topology=signature(Chem.Mol(item['mol_binary']))
        refs=item['references'];confs=entry['conformers']
        for ci,conf in enumerate(confs):
            molecule=conf['rd_mol'];positions=np.asarray(molecule.GetConformer().GetPositions())
            checks.append(dict(molecule_id=mid,reference_identity=rows[mid]['reference_identity'],conformer=ci,
                topology_order_matches=signature(molecule)==topology,
                coordinate_max_abs_error=float(np.max(np.abs(refs[ci].numpy()-positions))),
                upstream_matching_rmsd_field=conf.get('rmsd'),
                raw_archive_same_key_available=key in raw_map))
pd.DataFrame(checks).to_csv(OUT/'REFERENCE_ARCHIVE_CACHE_IDENTITY.csv',index=False)
frame=pd.DataFrame(checks)
result=dict(molecules=len(items),reference_conformers=len(frame),
    topology_order_mismatches=int((~frame.topology_order_matches).sum()),
    max_coordinate_error=float(frame.coordinate_max_abs_error.max()),
    conformers_with_matching_rmsd_marker=int(frame.upstream_matching_rmsd_field.notna().sum()),
    molecules_with_exact_raw_archive_key=int(frame.groupby('molecule_id').raw_archive_same_key_available.first().sum()),
    cached_reference_kind='torsional-diffusion standardized/matched RDKit conformers; not unmodified raw GEOM coordinates',
    upstream_source_evidence=dict(
      script=str(BASE/'external_repos/torsional-diffusion/standardize_confs.py'),
      seed_geometry='AllChem.EmbedMultipleConfs',optional_mmff_default=False,
      matching='rotatable-bond optimization, then conf[rd_mol]=mol_rdkit_single',
      uncertainty='archive generation command/options not independently recovered; geometry transformation source and matching rmsd markers corroborate standardized lineage'),
    interpretation='Reference V3D here characterizes this standardized pool, not the raw GEOM/xTB validity ceiling; do not infer raw physical ensemble variation from its BA spread')
(OUT/'REFERENCE_LINEAGE_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))

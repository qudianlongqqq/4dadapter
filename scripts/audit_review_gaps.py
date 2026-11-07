"""Bounded, reproducible exploratory audit; never overwrites frozen results."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, pickle, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.review_asset_paths import ASSET_ROOT,asset_path
ORIGINAL=ASSET_ROOT/'4dadapter-lsgoba-musigma-reliability-factorial'
DATA=ASSET_ROOT/'dataset/sixs_primary_final_evaluation_v1'
CART=ASSET_ROOT/'4dadapter-lsgoba-direct-cartesian-ablation'
PAPER=ASSET_ROOT/'4dadapter-lsgoba-final-directmu-ba'
SEEDS=(307,331,353)
BUDGETS=(0.004321542759325059,0.01,0.02)

def cache_for(out):
    return ROOT.parent/'review_gap_cache'/out.name

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def method_dirs():
    d={}
    for seed in SEEDS:
        d[f'full_{seed}']=DATA/f'methods/unrestricted_seed{seed}'
        d[f'noR_{seed}']=DATA.parent/f'sixs_final_matched_ablation_v1/methods/reliability_off_seed{seed}'
        d[f'cart_{seed}']=CART/f'artifacts/direct_cartesian_final_ablation/primary_final_2500x2/methods/direct_cartesian_seed{seed}'
    return d

def prepare(args):
    import numpy as np, pandas as pd, torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from etflow.ecir.j1r1_full_joint_unrestricted import UnrestrictedFullJointModel,unrestricted_full_joint_action
    from etflow.ecir.learned_geometry import geometry_values
    from etflow.ecir.review_interventions import primitive_direction,fixed_rms_proposal,rms,stereo_diagnostics
    torch.set_num_threads(2)
    protocol_path=ORIGINAL/'reports/ecir_mvr/sixs_primary_final_evaluation/00_FROZEN_FINAL_EVALUATION_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text())
    cache=json.loads((DATA/'topology_reference_cache/COMPLETE.json').read_text())
    assert cache['protocol_sha256']==sha(protocol_path)
    manifest=DATA.parent/'prospective_final_etflow_nfe10_seed42/SOURCE_RECORD_MANIFEST.jsonl'
    records=[json.loads(x) for x in manifest.read_text().splitlines() if x]
    assert sha(manifest)==protocol['bindings']['source_record_manifest_sha256']
    ids=sorted({r['canonical_identity'] for r in records},key=lambda s:hashlib.sha256(('review-audit-20261106|'+s).encode()).hexdigest())[:args.molecules]
    chosen=set(ids)
    frozen=dict(schema='pina-review-gap-audit-v1',status='FROZEN_BEFORE_NEW_INTERVENTION_OUTCOMES',
        role='exploratory reanalysis of previously evaluated primary-final cohort; not a new held-out test',
        selection='first N sha256(review-audit-20261106|molecule_id); both Sources retained',
        molecules=args.molecules,budgets_angstrom=BUDGETS,seeds=SEEDS,selected_molecule_ids=ids,
        primary_protocol_sha256=sha(protocol_path),source_manifest_sha256=sha(manifest),
        statistical_unit='molecule; average the two sources and then the three trained seeds',
        primary_budget_angstrom=0.01,bootstrap_replicates=10000,bootstrap_seed=20261106,
        planned_comparisons=['full-noR','full-cart','full-table_ba','full-table_bond','full-mmff_gradient','full-coefficient_sigma_stat'],
        no_training=True,no_checkpoint_selection=True,reference_used_only_for_evaluation=True,
        evaluation_precision='new coordinates rounded to SDF-equivalent four decimal places before V3D; report actual displacement too')
    if (args.out/'PROTOCOL.json').exists():
        assert json.loads((args.out/'PROTOCOL.json').read_text())==json.loads(json.dumps(frozen,default=str))
    else: write_json(args.out/'PROTOCOL.json',frozen)
    items=[]
    for path in cache['chunks']:
        data=torch.load(asset_path(path),map_location='cpu',weights_only=False)
        items.extend(i for i in data['items'] if i['molecule_id'] in chosen)
    assert len(items)==args.molecules
    items.sort(key=lambda i:ids.index(i['molecule_id']))
    wanted={r['record_id'] for i in items for r in i['source_records']}
    methods={}; provenance={}
    for name,directory in method_dirs().items():
        rows={}; files=[]
        for path in sorted((directory/'chunks').glob('*.pt')):
            local=torch.load(path,map_location='cpu',weights_only=False)
            rows.update({r['record_id']:r for r in local if r['record_id'] in wanted})
            files.append(dict(path=str(path),sha256=sha(path)))
        assert set(rows)==wanted,(name,len(rows),len(wanted))
        methods[name]=rows
        provenance[name]=dict(path=str(directory),files=files)
    cp=next(m for m in protocol['model_methods'] if m['id']=='unrestricted_seed307')
    checkpoint=ORIGINAL/cp['checkpoint']
    assert sha(checkpoint)==cp['checkpoint_sha256']
    model=UnrestrictedFullJointModel(128,3)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload.get('step',payload.get('optimizer_steps'))==17500
    model.load_state_dict(payload['model_state'],strict=True);model.eval()
    provenance['coefficient_intervention_checkpoint']=dict(path=str(checkpoint),sha256=sha(checkpoint))
    write_json(args.out/'INPUT_PROVENANCE.json',provenance)
    jobs=cache_for(args.out)/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    variability=[]; checks=[]; time_start=time.monotonic()
    for item_index,item in enumerate(items):
        jobpath=jobs/f'{item_index:04d}.pkl'
        graph=item['graph']; mol=Chem.Mol(item['mol_binary'])
        refs=item['references'].double()
        all_b,all_a=zip(*(geometry_values(x,graph) for x in refs))
        for fam,qs in [('bond',all_b),('angle_cos',all_a)]:
            qs=torch.stack(qs)
            for p in range(qs.shape[1]):
                variability.append(dict(molecule_id=item['molecule_id'],family=fam,primitive=p,
                    reference_conformers=len(refs),within_std=float(qs[:,p].std(unbiased=False)),
                    within_range=float(qs[:,p].max()-qs[:,p].min()),
                    train_table_scale=float((graph.bond_fixed if fam=='bond' else graph.angle_fixed)[p,1])))
        if jobpath.exists(): continue
        job=dict(molecule_id=item['molecule_id'],mol_binary=item['mol_binary'],records=[])
        with torch.inference_mode(): prediction=model.belief(graph)
        for row in item['source_records']:
            asset=asset_path(row['record_asset']);assert sha(asset)==row['record_asset_sha256']
            source=torch.as_tensor(torch.load(asset,map_location='cpu',weights_only=False)['source_coordinates'],dtype=torch.float64)
            rid=row['record_id']; cases=[('source',source)]
            # The same first Reference is evaluated for both Sources; all three
            # ensemble members are included and are never inference inputs.
            cases.extend((f'reference_{i}',r) for i,r in enumerate(refs))
            directions={}
            for name,lookup in methods.items():
                entry=lookup[rid]
                assert entry['molecule_id']==item['molecule_id']
                candidate=torch.as_tensor(entry['proposal_coordinates'],dtype=torch.float64)
                directions[name]=candidate-source
                cases.append((name+'_native',candidate))
            # Only analytic sigma coefficients change. Learned r and w are held
            # fixed at their unmodified full-model values; tau is budgeted below.
            with torch.inference_mode():
                act=unrestricted_full_joint_action(model,source,[graph],prediction)
                sigma_dir=primitive_direction(source,graph,prediction,
                    (act.bond_reliability,act.angle_reliability),act.family_weights[0],
                    coefficient_sigma=(graph.bond_fixed[:,1],graph.angle_fixed[:,1]))
            full_saved=torch.as_tensor(methods['full_307'][rid]['proposal_coordinates'],dtype=torch.float64)
            checks.append(dict(record_id=rid,max_abs_cpu_reproduction_error=float((act.proposal-full_saved).abs().max())))
            directions['coefficient_sigma_stat_307']=sigma_dir
            table=dict(bond_mu=graph.bond_fixed[:,0],bond_sigma=graph.bond_fixed[:,1],
                       angle_mu=graph.angle_fixed[:,0],angle_sigma=graph.angle_fixed[:,1])
            br=torch.ones(graph.bonds.shape[1]); ar=torch.ones(graph.angles.shape[0])
            for name,weights in [('table_ba',[.5,.5]),('table_bond',[1.,0.])]:
                directions[name]=primitive_direction(source,graph,table,(br,ar),weights)
            # A full MMFF94s force direction gives a cheap classical small-step
            # control at exactly the same Cartesian RMS, without target labels.
            native=Chem.Mol(mol); native.RemoveAllConformers(); conf=Chem.Conformer(mol.GetNumAtoms())
            for i,p in enumerate(source.tolist()):conf.SetAtomPosition(i,p)
            native.AddConformer(conf)
            mmff_supported=AllChem.MMFFHasAllMoleculeParams(native)
            if mmff_supported:
                props=AllChem.MMFFGetMoleculeProperties(native,mmffVariant='MMFF94s')
                ff=AllChem.MMFFGetMoleculeForceField(native,props,confId=0)
                directions['mmff_gradient']=-torch.tensor(ff.CalcGrad(),dtype=torch.float64).reshape(-1,3)
            else: directions['mmff_gradient']=torch.zeros_like(source)
            for name,direction in directions.items():
                for budget in BUDGETS:
                    cases.append((name+f'@{budget:.8f}',fixed_rms_proposal(source,direction,budget)))
            # Restraint strengths predeclared; no selecting a best one by V3D.
            for strength in (100.,1000.,10000.):
                restrained=Chem.Mol(native)
                try:
                    if not mmff_supported:raise ValueError('MMFF unsupported')
                    props=AllChem.MMFFGetMoleculeProperties(restrained,mmffVariant='MMFF94s')
                    ff=AllChem.MMFFGetMoleculeForceField(restrained,props,confId=0)
                    for i in range(mol.GetNumAtoms()):ff.MMFFAddPositionConstraint(i,0.,strength)
                    status=ff.Minimize(maxIts=50)
                    pos=torch.tensor(restrained.GetConformer().GetPositions(),dtype=torch.float64)
                except Exception: pos=source;status=-999
                cases.append((f'mmff_restrained_{strength:.0f}_status{status}',pos))
            case_rows=[]
            for name,pos in cases:
                before=float(rms(pos-source))
                rounded=torch.tensor(np.round(pos.numpy(),4),dtype=torch.float64)
                b,a=geometry_values(rounded,graph)
                rb,ra=all_b[0],all_a[0]
                case_rows.append(dict(case=name,coordinates=rounded.numpy(),
                    raw_rms_before_serialization=before,actual_rms=float(rms(rounded-source)),
                    max_atom_displacement=float(torch.linalg.vector_norm(rounded-source,dim=-1).max()),
                    bond_mae=float((b-rb).abs().mean()),angle_cos_mae=float((a-ra).abs().mean()),
                    **stereo_diagnostics(mol,source,rounded)))
            job['records'].append(dict(record_id=rid,cases=case_rows,mmff_supported=mmff_supported))
        with jobpath.open('wb') as f:pickle.dump(job,f,pickle.HIGHEST_PROTOCOL)
        if (item_index+1)%16==0:print('PREPARE',item_index+1,len(items),round(time.monotonic()-time_start,1),flush=True)
    pd.DataFrame(variability).to_csv(args.out/'REFERENCE_ENSEMBLE_VARIABILITY.csv',index=False)
    pd.DataFrame(checks).to_csv(args.out/'CPU_REPRODUCTION_CHECK.csv',index=False)
    write_json(args.out/'PREPARE_COMPLETE.json',dict(molecules=len(items),records=len(wanted),jobs=len(list(jobs.glob('*.pkl')))))


def evaluate(args):
    import numpy as np,pandas as pd,types
    from rdkit import Chem
    # Use the exact locally pinned official evaluator and frozen reference cache.
    adapter=ASSET_ROOT/'4dadapter-lsgo-v2/scripts/run_lsgo_standard_genbench3d.py'
    assert sha(adapter)=='6ebc450b4f841a9f6a3b463b7838e50bc2951e92570146cd0a473abbfd970450'
    reference_root=ASSET_ROOT/'4dadapter-lsgo-v2/reports/ecir_mvr/lsgo_standard_eval/genbench3d_reference_cache'
    for filename,expected in [('LigBoundConf_geometry_values.p','63659acddd04017a4b8fc5f2df767540e48cd36a7849e578ddb6caf6130deadc'),
                              ('LigBoundConf_geometry_kernel_densities.p','6c098fa5b10c85f12db49df3a35efa33963fc222754e5d2a9d0b64e61c604a19')]:
        assert sha(reference_root/filename)==expected
    sys.modules.setdefault('gemmi',types.ModuleType('gemmi'))
    spec=importlib.util.spec_from_file_location('review_genbench_adapter',adapter)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    official.GENBENCH_REPO=Path(os.environ.get('PINA_GENBENCH_REPO',str(official.GENBENCH_REPO)))
    official.LIGBOUNDCONF=ASSET_ROOT/'external_data/genbench3d_official/ligboundconf_minimized/S2_LigBoundConf_minimized.sdf'
    official.REFERENCE_ROOT=reference_root
    SDFSource,ReferenceGeometry,Validity3D=official.import_official_evaluator()
    reference=ReferenceGeometry(source=SDFSource(str(official.LIGBOUNDCONF),name='LigBoundConf',removeHs=False),
        root=str(official.REFERENCE_ROOT),minimum_pattern_values=50,use_generalized_patterns=False)
    validity=Validity3D(reference_geometry=reference,q_value_threshold=.001,steric_clash_safety_ratio=.75,
        maximum_ring_plane_distance=.1,include_torsions=False,consider_hydrogens=False)
    dest=cache_for(args.out)/'evaluated';dest.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    for ji,path in enumerate(sorted((cache_for(args.out)/'jobs').glob('*.pkl'))):
        result_path=dest/(path.stem+'.parquet')
        if result_path.exists(): continue
        with path.open('rb') as f:job=pickle.load(f)
        mol=Chem.Mol(job['mol_binary']);mol.RemoveAllConformers();flat=[]
        for row in job['records']:
            for case in row['cases']:
                conf=Chem.Conformer(mol.GetNumAtoms())
                for i,p in enumerate(case['coordinates']):conf.SetAtomPosition(i,p)
                mol.AddConformer(conf,assignId=True)
                flat.append(dict(molecule_id=job['molecule_id'],record_id=row['record_id'],
                    mmff_supported=row['mmff_supported'],**{k:v for k,v in case.items() if k!='coordinates'}))
        try:
            bs,_,_=validity.analyze_bonds(mol); ass,_,_=validity.analyze_angles(mol)
            rings,_=validity.analyze_rings(mol);clashes,_=validity.analyze_clashes(mol)
            sets=[set(bs),set(ass),set(rings),set(clashes)]
            for i,row in enumerate(flat):
                passes=[i in s for s in sets]
                row.update(dict(zip(['bond_valid','angle_valid','ring_valid','clash_valid'],passes)))
                row.update(validity3d=all(passes),evaluator_success=True,error='')
        except Exception as exc:
            # Fail fixed denominator; never silently drop a molecule or method.
            for row in flat:row.update(validity3d=False,bond_valid=False,angle_valid=False,
                ring_valid=False,clash_valid=False,evaluator_success=False,error=repr(exc))
        pd.DataFrame(flat).to_parquet(result_path,index=False)
        print('EVALUATED',ji+1,'seconds',round(time.monotonic()-start,1),'rows',len(flat),flush=True)
    files=sorted(dest.glob('*.parquet'))
    frame=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    frame.to_parquet(args.out/'PER_RECORD_INTERVENTIONS.parquet',index=False)
    print('TOTAL_ROWS',len(frame),'FAILURES',int((~frame.evaluator_success).sum()),flush=True)


def aggregate(args):
    import numpy as np,pandas as pd
    frame=pd.read_parquet(args.out/'PER_RECORD_INTERVENTIONS.parquet')
    # Optimizer convergence is a diagnostic field, not a separate method.
    frame['case']=frame.case.str.replace(r'_status-?\d+$','',regex=True)
    assert not frame.duplicated(['record_id','case']).any()
    columns=['validity3d','bond_valid','angle_valid','ring_valid','clash_valid','actual_rms',
        'max_atom_displacement','bond_mae','angle_cos_mae','evaluator_success']
    summary=frame.groupby('case')[columns].mean()
    summary['records']=frame.groupby('case').size()
    summary.to_csv(args.out/'INTERVENTION_SUMMARY.csv')
    stereo_cols=[c for c in frame if c.startswith(('tetra_','ez_'))]
    frame.groupby('case')[stereo_cols].sum().to_csv(args.out/'STEREO_SUMMARY.csv')
    native=frame[frame.case.str.endswith('_native')]
    native.groupby('case')['max_atom_displacement'].quantile([.5,.95,.99,1]).to_csv(args.out/'NATIVE_MOVEMENT_TAILS.csv')
    # Pair within molecule and seed before bootstrapping molecules.
    comparisons=[];rng=np.random.default_rng(20261106)
    for budget in BUDGETS:
        suffix=f'@{budget:.8f}'
        for comp in ['noR','cart','table_ba','table_bond','mmff_gradient','coefficient_sigma_stat']:
            seeds=SEEDS if comp in ['noR','cart'] else ((307,) if comp=='coefficient_sigma_stat' else SEEDS)
            per_seed=[]
            for seed in seeds:
                left=f'full_{seed}{suffix}'
                right=(f'{comp}_{seed}' if comp in ['noR','cart','coefficient_sigma_stat'] else comp)+suffix
                a=frame[frame.case==left].set_index('record_id')
                b=frame[frame.case==right].set_index('record_id')
                assert set(a.index)==set(b.index)
                b=b.loc[a.index]
                delta=a.validity3d.astype(float)-b.validity3d.astype(float)
                local=pd.DataFrame(dict(molecule_id=a.molecule_id,delta=delta)).groupby('molecule_id').delta.mean()
                per_seed.append(local)
            values=pd.concat(per_seed,axis=1).mean(axis=1).to_numpy()
            boot=np.concatenate([values[rng.integers(len(values),size=(1000,len(values)))].mean(1) for _ in range(10)])
            comparisons.append(dict(comparator=comp,budget=budget,molecules=len(values),seed_count=len(seeds),
                delta_v3d_pp=100*values.mean(),ci_low_pp=100*np.quantile(boot,.025),ci_high_pp=100*np.quantile(boot,.975)))
    pd.DataFrame(comparisons).to_csv(args.out/'PAIRED_MOLECULE_BOOTSTRAP.csv',index=False)
    # Control reproduction against old saved V3D outcomes; not used for selection.
    parity=[]
    for name,directory in method_dirs().items():
        old=pd.read_parquet(directory/'VALIDITY3D.parquet').set_index('record_id')
        new=frame[frame.case==name+'_native'].set_index('record_id')
        old=old.loc[new.index]
        parity.append(dict(method=name,records=len(new),mismatches=int((old.validity3d.astype(bool)!=new.validity3d).sum()),
                           old_pass_rate=float(old.validity3d.mean()),new_pass_rate=float(new.validity3d.mean())))
    pd.DataFrame(parity).to_csv(args.out/'NATIVE_EVALUATOR_PARITY.csv',index=False)
    refs=frame[frame.case.str.startswith('reference_')].drop_duplicates(['molecule_id','case'])
    refs.groupby('case')[columns].mean().to_csv(args.out/'REFERENCE_VALIDITY.csv')
    refs.groupby('molecule_id')[columns].mean().mean().to_csv(args.out/'REFERENCE_VALIDITY_MOLECULE_MACRO.csv')
    var=pd.read_csv(args.out/'REFERENCE_ENSEMBLE_VARIABILITY.csv')
    var['within_std_over_train_scale']=var.within_std/var.train_table_scale
    var.groupby('family')[['within_std','within_range','within_std_over_train_scale']].quantile([.5,.9,.99]).to_csv(args.out/'REFERENCE_VARIABILITY_QUANTILES.csv')
    write_json(args.out/'FINAL_STATUS.json',dict(status='COMPLETE',molecules=frame.molecule_id.nunique(),records=frame.record_id.nunique(),
        cases=frame.case.nunique(),evaluated_rows=len(frame),evaluator_failed_rows=int((~frame.evaluator_success).sum()),
        native_parity_mismatches=sum(p['mismatches'] for p in parity),training_run=False,production_model_modified=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def census(args):
    import pandas as pd,numpy as np,torch
    from rdkit import Chem
    from etflow.ecir.review_interventions import stereo_diagnostics
    torch.set_num_threads(1)
    source_path=DATA/'methods/source/COORDINATES.sdf'
    source={}
    for mol in Chem.SDMolSupplier(str(source_path),removeHs=False,sanitize=True):
        if mol is None:raise ValueError('Source SDF parse failure')
        rid=mol.GetProp('_Name');assert rid not in source
        source[rid]=(mol,np.array(mol.GetConformer().GetPositions()))
    assert len(source)==5000
    rows=[];components=[]
    for seed in SEEDS:
        directory=DATA/f'methods/unrestricted_seed{seed}'
        for mol in Chem.SDMolSupplier(str(directory/'COORDINATES.sdf'),removeHs=False,sanitize=True):
            if mol is None:raise ValueError('Output SDF parse failure')
            rid=mol.GetProp('_Name');smol,s=source[rid]
            assert tuple(a.GetAtomicNum() for a in mol.GetAtoms())==tuple(a.GetAtomicNum() for a in smol.GetAtoms())
            x=np.array(mol.GetConformer().GetPositions());delta=x-s
            rows.append(dict(seed=seed,record_id=rid,raw_rms=float(np.sqrt(np.sum(delta*delta,axis=1).mean())),
                max_atom_displacement=float(np.linalg.norm(delta,axis=1).max()),**stereo_diagnostics(smol,s,x)))
        print('CENSUS',seed,len(rows),flush=True)
        v=pd.read_parquet(directory/'VALIDITY3D.parquet').set_index('record_id')
        sv=pd.read_parquet(DATA/'methods/source/VALIDITY3D.parquet').set_index('record_id').loc[v.index]
        for col in ['validity3d','bond_geometry_valid','angle_geometry_valid','aromatic_ring_valid','intramolecular_steric_clash_valid']:
            a,b=sv[col].astype(bool),v[col].astype(bool)
            components.append(dict(seed=seed,component=col,records=len(v),source_rate=a.mean(),output_rate=b.mean(),
                repaired=int((~a&b).sum()),damaged=int((a&~b).sum())))
    frame=pd.DataFrame(rows);assert len(frame)==15000
    frame.to_parquet(args.out/'FULL_COHORT_STEREO_PER_RECORD.parquet',index=False)
    cols=[c for c in frame if c.startswith(('tetra_','ez_'))]
    frame.groupby('seed')[cols].sum().to_csv(args.out/'FULL_COHORT_STEREO_COUNTS.csv')
    frame.groupby('seed')[['raw_rms','max_atom_displacement']].quantile([.5,.95,.99,1]).to_csv(args.out/'FULL_COHORT_MOVEMENT_QUANTILES.csv')
    pd.DataFrame(components).to_csv(args.out/'FULL_COHORT_V3D_COMPONENT_TRANSITIONS.csv',index=False)
    write_json(args.out/'FULL_COHORT_CENSUS_STATUS.json',dict(status='COMPLETE',records_per_seed=5000,molecules=2500,
        seeds=SEEDS,stereo_scope='Source-specified tetrahedral/EZ coordinate preservation, not agreement with chemical target',
        evaluated_coordinates='frozen SDF coordinates; no model inference'))


def guard(args):
    import numpy as np,torch
    from rdkit import Chem
    from etflow.ecir.review_interventions import bounded_stereo_proposal,rms,stereo_diagnostics
    torch.set_num_threads(1)
    out=args.out/'guard_variant';out.mkdir(exist_ok=True);(cache_for(out)/'jobs').mkdir(parents=True,exist_ok=True)
    write_json(out/'PROTOCOL.json',dict(role='exploratory new inference variant after discovery of unrestricted movement tails',
        selected_cohort='same 256 molecule hash subset as parent protocol',nominal_atom_caps=[.03,.1],
        max_rms=.03,rounding_margin=.0002,source_labels_or_v3d_used_for_acceptance=False,
        guard='uniform shrink plus Source-specified tetrahedral/EZ preservation with backtracking',
        freeze='before guarded variant V3D outcomes; caps not selected by performance'))
    for path in sorted((cache_for(args.out)/'jobs').glob('*.pkl')):
        with path.open('rb') as f:job=pickle.load(f)
        mol=Chem.Mol(job['mol_binary']);recs=[]
        for record in job['records']:
            source=torch.tensor(next(c['coordinates'] for c in record['cases'] if c['case']=='source'),dtype=torch.float64)
            native=[c for c in record['cases'] if c['case']=='source' or (c['case'].startswith('full_') and c['case'].endswith('_native'))]
            cases=[dict(c) for c in native]
            for original in native:
                if original['case']=='source':continue
                for cap in (.03,.1):
                    pos,meta=bounded_stereo_proposal(mol,source,torch.tensor(original['coordinates']),
                        max_atom_displacement=cap-.0002,max_rms=.03-.0002)
                    rounded=torch.tensor(np.round(pos.numpy(),4),dtype=torch.float64)
                    assert float(torch.linalg.vector_norm(rounded-source,dim=-1).max())<=cap
                    assert float(rms(rounded-source))<=.03
                    diag=stereo_diagnostics(mol,source,rounded)
                    assert not (diag['tetra_flips'] or diag['ez_flips'] or diag['tetra_candidate_degenerate'] or diag['ez_candidate_degenerate'])
                    cases.append(dict(case=original['case'].replace('_native',f'_guard{cap:.2f}'),coordinates=rounded.numpy(),
                        raw_rms_before_serialization=float(rms(pos-source)),actual_rms=float(rms(rounded-source)),
                        max_atom_displacement=float(torch.linalg.vector_norm(rounded-source,dim=-1).max()),
                        scale=meta['scale'],guard_reason=meta['reason'],backtracks=meta['backtracks'],**diag))
            recs.append(dict(record_id=record['record_id'],mmff_supported=record['mmff_supported'],cases=cases))
        job['records']=recs
        with (cache_for(out)/'jobs'/path.name).open('wb') as f:pickle.dump(job,f,pickle.HIGHEST_PROTOCOL)
    print('GUARD_JOBS_READY',len(list((cache_for(out)/'jobs').glob('*.pkl'))),flush=True)


def guard_summary(args):
    import pandas as pd,numpy as np
    out=args.out/'guard_variant'
    f=pd.read_parquet(out/'PER_RECORD_INTERVENTIONS.parquet')
    cols=['validity3d','bond_valid','angle_valid','ring_valid','clash_valid','actual_rms','max_atom_displacement']
    summary=f.groupby('case')[cols].mean();summary['max_displacement_global']=f.groupby('case').max_atom_displacement.max()
    summary['changed_fraction']=f.assign(changed=f.scale.fillna(1.)<1).groupby('case').changed.mean()
    summary.to_csv(out/'GUARD_SUMMARY.csv')
    comparisons=[];rng=np.random.default_rng(20261106)
    for cap in (.03,.1):
        deltas=[]
        for seed in SEEDS:
            a=f[f.case==f'full_{seed}_native'].set_index('record_id')
            b=f[f.case==f'full_{seed}_guard{cap:.2f}'].set_index('record_id').loc[a.index]
            deltas.append(pd.DataFrame(dict(molecule_id=a.molecule_id,delta=b.validity3d.astype(float)-a.validity3d.astype(float))).groupby('molecule_id').delta.mean())
        values=pd.concat(deltas,axis=1).mean(axis=1).to_numpy()
        boot=np.concatenate([values[rng.integers(len(values),size=(1000,len(values)))].mean(1) for _ in range(10)])
        comparisons.append(dict(atom_cap=cap,delta_guard_minus_native_pp=100*values.mean(),
            ci_low_pp=100*np.quantile(boot,.025),ci_high_pp=100*np.quantile(boot,.975)))
    pd.DataFrame(comparisons).to_csv(out/'GUARD_PAIRED_BOOTSTRAP.csv',index=False)
    write_json(out/'FINAL_STATUS.json',dict(status='COMPLETE',molecules=f.molecule_id.nunique(),records=f.record_id.nunique(),
        evaluated_rows=len(f),failed_rows=int((~f.evaluator_success).sum()),production_variant_replaced=False))
    print(summary.to_string());print(comparisons)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','aggregate','census','guard','guard_summary'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--molecules',type=int,default=256)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'evaluate':evaluate,'aggregate':aggregate,'census':census,'guard':guard,'guard_summary':guard_summary}[a.stage](a)

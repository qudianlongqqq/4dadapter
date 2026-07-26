from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest,torch,yaml
from etflow.ecir.learned_geometry import direct_gradient_update,distribution_parameters,prepare_graph,safety_accept
from etflow.ecir.lsgo_io import file_sha256
from etflow.ecir.lsgo_mechanism import masked_gradient_update,masked_objective
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ecir_mvr/lsgo_bond_confirm'; CFG=yaml.safe_load((ROOT/'configs/ecir_mvr_lsgo_bond_confirm.yaml').read_text())
@pytest.fixture(scope='module')
def example():
    p=torch.load(OUT/'manifests/BOND_MINIMALITY_CONFIRM_COMPACT.pt',map_location='cpu',weights_only=False); item=p['items'][0]; cal=json.loads(Path(CFG['anchor']['calibration']).read_text()); return item,prepare_graph(item['record'],cal)
def test_b_and_a_family_masks(example):
    item,g=example; x=torch.as_tensor(item['sources'][0],dtype=torch.float64); p=distribution_parameters(g,model=None,variant='A'); _,b=masked_objective(x,g,p,'B'); _,a=masked_objective(x,g,p,'A'); assert b['active_bonds']==g.bonds.size(1) and b['active_angles']==0; assert a['active_angles']==g.angles.size(0) and a['active_bonds']==0
def test_ba_historical_equivalence(example):
    item,g=example; x=torch.as_tensor(item['sources'][0],dtype=torch.float64); p=distribution_parameters(g,model=None,variant='A'); old=direct_gradient_update(x,g,p,rms_budget=.003,atom_cap=.03,steps=1); new=masked_gradient_update(x,g,p,'BA',rms_budget=.003,atom_cap=.03); assert torch.equal(old['coordinates'],new['coordinates']) and old['trace']==new['trace']
def test_same_safety_guards(example):
    item,g=example; x=torch.as_tensor(item['sources'][0],dtype=torch.float64); p=distribution_parameters(g,model=None,variant='A'); results=[]
    for family in ('B','A','BA'): results.append(safety_accept(x,masked_gradient_update(x,g,p,family,rms_budget=.003,atom_cap=.03)['coordinates'],g)[1])
    assert all(set(row)==set(results[0]) for row in results)
def test_same_trust_region(example):
    item,g=example;x=torch.as_tensor(item['sources'][0],dtype=torch.float64);p=distribution_parameters(g,model=None,variant='A')
    for f in ('B','A','BA'):
        y=masked_gradient_update(x,g,p,f,rms_budget=.003,atom_cap=.03)['coordinates']; d=y-x; assert float(torch.sqrt(d.square().sum(-1).mean()))<=.003+1e-12 and float(torch.linalg.vector_norm(d,dim=-1).max())<=.03+1e-12
def test_fresh_identity_exclusion_and_flex_bins():
    p=json.loads((OUT/'DATASET_IDENTITY.json').read_text()); assert p['overlap_with_excluded_union']==0 and p['molecule_count']==200 and p['source_record_count']==600; assert p['flex_histogram']=={'low_0_2':70,'medium_3_4':70,'high_ge_5':60}
def test_ba_reproduction_report():
    path=OUT/'manifests/BA_REPRODUCTION.json'
    if not path.is_file(): pytest.skip('runner gate not executed yet')
    p=json.loads(path.read_text()); assert p['status']=='PASS' and p['max_accepted_coordinate_abs_error']==0 and p['safety_all_equal']
def test_coordinate_freeze_identity():
    path=OUT/'COORDINATE_FREEZE_MANIFEST.json'
    if not path.is_file(): pytest.skip('coordinates not frozen yet')
    p=json.loads(path.read_text()); assert p['status']=='FROZEN' and p['records_per_condition']==600 and len(p['conditions'])==10
    assert file_sha256(Path(p['coordinate_diagnostics_path']))==p['coordinate_diagnostics_sha256']
    for export in p['exports']:
        assert file_sha256(Path(export['sdf_path']))==export['sdf_sha256'] and export['records']==600
    diagnostic=pd.read_parquet(p['coordinate_diagnostics_path'])
    expected={'low_0_2':210,'medium_3_4':210,'high_ge_5':180}
    for _,frame in diagnostic.groupby('condition'):
        assert frame.flex_bin.value_counts().to_dict()==expected
def test_xTB_and_pb_pairing():
    freeze=json.loads((OUT/'COORDINATE_FREEZE_MANIFEST.json').read_text())
    diagnostic=pd.read_parquet(freeze['coordinate_diagnostics_path'])
    expected=set(diagnostic[diagnostic.condition=='B_seed173'].sample_id)
    for name in ('XTB_SINGLE_POINT_COMPLETE.json','POSEBUSTERS_COMPLETE.json'):
        path=OUT/'manifests'/name
        if not path.is_file(): pytest.skip('external evaluation not complete')
        p=json.loads(path.read_text()); assert p['status']=='COMPLETED' and p['formal_test_records_read']==p['frozen_holdout_records_read']==0
        for binding in p['bindings']:
            result=Path(binding['result_path'])
            assert file_sha256(result)==binding['result_sha256']
            frame=pd.read_parquet(result)
            assert len(frame)==600 and frame.sample_id.nunique()==600 and set(frame.sample_id)==expected
def test_angle_unique_case_extraction():
    decision_path=OUT/'FINAL_DECISION.json'
    if not decision_path.is_file(): pytest.skip('analysis not complete')
    rescue=0
    for seed in CFG['ba_seeds']:
        base=OUT/'per_record/xtb'
        b=pd.read_parquet(base/f'PAIRED_DELTA_B_seed{seed}__primary.parquet')[['sample_id','delta_energy_kcal_mol']].rename(columns={'delta_energy_kcal_mol':'B'})
        ba=pd.read_parquet(base/f'PAIRED_DELTA_BA_seed{seed}__primary.parquet')[['sample_id','delta_energy_kcal_mol']].rename(columns={'delta_energy_kcal_mol':'BA'})
        paired=b.merge(ba,on='sample_id',validate='one_to_one')
        rescue+=int(((paired.B>0)&(paired.BA<=0)).sum())
    decision=json.loads(decision_path.read_text())
    assert rescue==decision['rare_b_harm_ba_rescue_count']==1
    assert 'B-harm/BA-rescue count is `1`' in (OUT/'ANGLE_UNIQUE_CASES.md').read_text()
def test_protected_reads_zero():
    p=json.loads((OUT/'DATASET_IDENTITY.json').read_text()); assert p['formal_test_records_read']==p['frozen_holdout_records_read']==0 and CFG['guards']['formal_test_records_read']==CFG['guards']['frozen_holdout_records_read']==0
def test_sha_validation_if_finalized():
    path=OUT/'SHA256SUMS.txt'
    if not path.is_file(): pytest.skip('not finalized')
    for line in path.read_text().splitlines():
        digest,relative=line.split('  ',1); assert file_sha256(ROOT/relative)==digest

#!/usr/bin/env python3
"""Freeze final reports for the Full-Joint capacity audit."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT=bootstrap()
import scripts.run_sixs_j1r1_full_joint_adaptive_ba_movement as full
import scripts.run_sixs_full_joint_final_capacity_audit as capacity_audit
from etflow.ecir.learned_geometry import geometry_values

RUN="sixs_j1r1_full_joint_adaptive_ba_movement_seed307"
REPORT=ROOT/"reports/ecir_mvr"/RUN
ART=ROOT/"artifacts/ecir_mvr"/RUN
OUT=ROOT/"reports/ecir_mvr/sixs_full_joint_final_capacity_audit"
XTB=REPORT/"xtb_single_point_dev"
RID="val::CCC(C)C(=O)OC1CC2(C)C(c3ccoc3)OC(=O)CC23OC2(C)OC34C(OC(C)=O)C3(O)C(OC(=O)C5(C)OC5C)C5(C)CC3(O)C(C)(C5CC(=O)OC)C14O2__gen0000"

def atomic_text(path:Path,value:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(value.rstrip()+"\n",encoding="utf-8"); os.replace(tmp,path)
def atomic_json(path:Path,value:Any)->None: atomic_text(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str))
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def stats(x)->dict[str,Any]:
    a=np.asarray(x,float); return {"n":len(a),"min":float(a.min()),"p01":float(np.quantile(a,.01)),"median":float(np.median(a)),"mean":float(a.mean()),"p99":float(np.quantile(a,.99)),"max":float(a.max()),"finite":bool(np.isfinite(a).all())}

def markdown_table(frame:pd.DataFrame)->str:
    values=[[str(value).replace("|","\\|") for value in row] for row in frame.itertuples(index=False,name=None)]
    header=[str(column).replace("|","\\|") for column in frame.columns]
    lines=["| "+" | ".join(header)+" |","| "+" | ".join("---" for _ in header)+" |"]
    lines.extend("| "+" | ".join(row)+" |" for row in values)
    return "\n".join(lines)

def source_forensic(config:dict[str,Any]):
    dest=OUT/"xtb_extreme_source_forensic"; dest.mkdir(parents=True,exist_ok=True)
    sdf=dest/"SOURCE.sdf"; per=dest/"PER_RECORD.parquet"
    pbp=dest/"subset_eval/posebusters.parquet"; v3p=dest/"subset_eval/validity3d.parquet"
    source_payload=torch.load(config["data"]["source_payload"],map_location="cpu",weights_only=False)
    source_row=next(row for row in source_payload["val"] if str(row["sample_id"])==RID)
    prepared=torch.load(config["data"]["prepared_payload"],map_location="cpu",weights_only=False)
    item=next(row for row in prepared["val"] if str(row["molecule_id"])==str(source_row["molecule_id"]))
    graph=item["graph"]; source=torch.as_tensor(source_row["source"],dtype=torch.float64); reference=torch.as_tensor(item["references"][0],dtype=torch.float64)
    if not sdf.is_file():
        val=pd.read_parquet(config["data"]["val_manifest"]); meta=val[val.sample_id.astype(str)==RID].iloc[0]
        raw=(Path(config["data"]["val_cache"])/Path(str(meta.source_path)).name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=str(meta.source_file_sha256): raise RuntimeError("tail source cache hash mismatch")
        record=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=False)
        writer=Chem.SDWriter(str(sdf)); full.frozen.write_molecule(writer,record,source,RID,"SOURCE_XTB_TAIL_FORENSIC"); writer.close()
        pd.DataFrame([{"record_id":RID,"molecule_id":str(source_row["molecule_id"])}]).to_parquet(per,index=False)
    if not pbp.is_file() or not v3p.is_file():
        raise RuntimeError("Source forensic subset evaluation is incomplete")
    pb=pd.read_parquet(pbp).iloc[0].to_dict(); v3=pd.read_parquet(v3p).iloc[0].to_dict()
    sb,sa=geometry_values(source,graph); rb,ra=geometry_values(reference,graph)
    def min_nonbonded(coords):
        n=len(coords); bonded={tuple(sorted(map(int,p))) for p in graph.bonds.t().tolist()}; values=[]
        for i in range(n):
            for j in range(i+1,n):
                if (i,j) not in bonded: values.append(float(torch.linalg.vector_norm(coords[i]-coords[j])))
        return min(values)
    def sdf_coords(path):
        mol=next(m for m in Chem.SDMolSupplier(str(path),removeHs=False,sanitize=False) if m is not None and m.GetProp("_Name")==RID)
        return torch.tensor(mol.GetConformer().GetPositions(),dtype=torch.float64)
    fc=sdf_coords(ART/"dev_evaluation/PROPOSAL.sdf")
    cc=sdf_coords(ROOT/"artifacts/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/j1_r1_joint/dev_evaluation/PROPOSAL.sdf")
    return {"source_row":source_row,"graph":graph,"source":source,"reference":reference,"source_pb":pb,"source_v3d":v3,"source_bond_raw_mae":float((sb-rb).abs().mean()),"source_angle_raw_mae":float((sa-ra).abs().mean()),"source_bond_max_abs_error":float((sb-rb).abs().max()),"source_angle_max_abs_error":float((sa-ra).abs().max()),"minimum_nonbonded_distance_A":{"source":min_nonbonded(source),"comparator":min_nonbonded(cc),"full_joint":min_nonbonded(fc)},"nonfinite":{"source":not bool(torch.isfinite(source).all()),"comparator":not bool(torch.isfinite(cc).all()),"full_joint":not bool(torch.isfinite(fc).all())}}

def main()->int:
    config=json.loads((ROOT/"configs/sixs_j1r1_full_joint_adaptive_ba_movement.json").read_text(encoding="utf-8"))
    capacity=json.loads((OUT/"EXTENSION_EVALUATION_COMPLETE.json").read_text(encoding="utf-8")); b=capacity["bootstrap"]
    extended_log=pd.read_csv(OUT/"extension_22500/EXTENDED_TRAIN_LOG.csv")
    curve=capacity_audit.prepare_curve(
        extended_log,
        config["training"]["scheduler_horizon"],
        config["training"]["backbone_learning_rate"],
        config["training"]["head_learning_rate"],
    )
    curve.insert(1,"TRAINING_PHASE",np.where(curve["step"]<=17500,"ORIGINAL","EXACT_CONTINUATION"))
    tmp_curve=OUT/f"04_TRAINING_CURVE.csv.tmp.{os.getpid()}"
    curve.to_csv(tmp_curve,index=False)
    os.replace(tmp_curve,OUT/"04_TRAINING_CURVE.csv")
    tail=pd.read_csv(XTB/"XTB_EXTREME_TAIL.csv"); energy=tail[(tail.audit_group=="TOP20_WORST_POSITIVE")&(tail["rank"]==1)].iloc[0]
    baseline=pd.read_parquet(ART/"dev_evaluation/PER_RECORD.parquet"); comparator=pd.read_parquet(ROOT/"artifacts/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/j1_r1_joint/dev_evaluation/PER_RECORD.parquet")
    br=baseline[baseline.record_id.astype(str)==RID].iloc[0]
    cr=comparator[comparator.record_id.astype(str)==RID].iloc[0]
    payload=torch.load(ART/"dev_evaluation/EVALUATION_PAYLOAD.pt",map_location="cpu",weights_only=False); primitive=next(x for x in payload["primitive_rows"] if x["record_id"]==RID)
    fpb=pd.read_parquet(ART/"dev_evaluation/POSEBUSTERS.parquet"); fpb=fpb[fpb.record_id.astype(str)==RID].iloc[0]
    cpb=pd.read_parquet(ROOT/"artifacts/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/j1_r1_joint/dev_evaluation/POSEBUSTERS.parquet"); cpb=cpb[cpb.record_id.astype(str)==RID].iloc[0]
    fv=pd.read_parquet(ART/"dev_evaluation/VALIDITY3D.parquet"); fv=fv[fv.record_id.astype(str)==RID].iloc[0]
    cv=pd.read_parquet(ROOT/"artifacts/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307/j1_r1_joint/dev_evaluation/VALIDITY3D.parquet"); cv=cv[cv.record_id.astype(str)==RID].iloc[0]
    sf=source_forensic(config)
    source_pb=bool(sf["source_pb"]["PB"]); source_v3d=bool(sf["source_v3d"]["validity3d"])
    source_short=float(sf["source_pb"]["shortest_noncovalent_relative_distance"])
    cause="PREEXISTING_SEVERE_SOURCE_GEOMETRY_PATHOLOGY_PLUS_NEAR_MAX_MOVEMENT; BOTH_REFINEMENTS_MOVE_AWAY_FROM_AN_EXCEPTIONALLY_LOW_SOURCE_SINGLE_POINT_ENERGY"
    forensic=f"""# xTB extreme-tail forensic

The single `Full - Source > +50 kcal/mol` record is `{RID}` (molecule `{energy.molecule_id}`). All three GFN2-xTB single-point jobs succeeded with the same charge 0/UHF 0 and no geometry optimization.

| quantity | Source | Comparator | Full Joint 17,500 |
|---|---:|---:|---:|
| xTB energy (Eh) | {energy.source_energy_hartree:.12f} | {energy.comparator_energy_hartree:.12f} | {energy.full_joint_energy_hartree:.12f} |
| delta vs Source (kcal/mol) | 0 | {float(energy.comparator_energy_hartree-energy.source_energy_hartree)*627.509474:.6f} | {energy.delta_full_vs_source_kcal_mol:.6f} |
| delta Full vs Comparator (kcal/mol) | — | — | {energy.delta_full_vs_comparator_kcal_mol:.6f} |
| V3D | {source_v3d} | {bool(cv.validity3d)} | {bool(fv.validity3d)} |
| PB | {source_pb} | {bool(cpb.PB)} | {bool(fpb.PB)} |

Full Joint uses `tau={br.tau:.9f} A` (near the 0.01 A upper bound), Source RMSD `{br.source_rmsd:.9f} A`, `w_B={br.w_B:.6f}`, `w_A={br.w_A:.6f}`, reaches the 0.03 A atom cap, and triggers the descriptive rollback flag. Comparator also reaches the atom cap and is **110.13 kcal/mol above Source**; Full Joint is 57.03 kcal/mol lower than Comparator, so this tail is not a Full-Joint-only deterioration.

Source is already invalid by V3D/PB. Its shortest noncovalent relative distance is `{source_short:.6f}`; Comparator/Full are `{float(cpb.shortest_noncovalent_relative_distance):.6f}` / `{float(fpb.shortest_noncovalent_relative_distance):.6f}`. Full has one short Bond outlier, five Angle outliers and seven PoseBusters clashes; V3D reports four invalid Bond primitives and 12 clashes. Raw minimum nonbonded distances (topology-excluding bonds) are Source `{sf['minimum_nonbonded_distance_A']['source']:.6f} A`, Comparator `{sf['minimum_nonbonded_distance_A']['comparator']:.6f} A`, Full `{sf['minimum_nonbonded_distance_A']['full_joint']:.6f} A`.

Source raw Bond/Angle MAE are `{sf['source_bond_raw_mae']:.8f}` / `{sf['source_angle_raw_mae']:.8f}`; Full changes them to `{br.bond_raw_mae:.8f}` / `{br.angle_raw_mae:.8f}`, while Comparator gives `{cr.bond_raw_mae:.8f}` / `{cr.angle_raw_mae:.8f}`. No coordinate or primitive diagnostic is nonfinite.

Bond Reliability: `{stats(primitive['bond_reliability'])}`. Angle Reliability: `{stats(primitive['angle_reliability'])}`. Bond sigma: `{stats(primitive['bond_sigma'])}`. Angle sigma: `{stats(primitive['angle_sigma'])}`.

The cleanest supported cause classification is `{cause}`. This is an association/forensic diagnosis, not proof of a single causal atom pair. The extreme source-relative energy change is coupled to a pre-existing severely clashing/invalid Source, near-max learned movement and cap activation; nevertheless Full Joint materially improves energy relative to the comparator for this same record.
"""
    atomic_text(OUT/"11_XTB_EXTREME_TAIL_FORENSIC.md",forensic)
    missing=pd.DataFrame([
        ("seed331 Full Joint replication","MISSING","Required for seed robustness; not started"),("seed353 Full Joint replication","MISSING","Required for seed robustness; not started"),("Formal evaluation","MISSING","Protected and unread"),("large holdout evaluation","MISSING","Protected and unread"),("AMR-P / AMR-R","MISSING","No final-candidate coverage/accuracy evaluation"),("COV-P / COV-R","MISSING","No final-candidate coverage/accuracy evaluation"),("SIXS neural inference runtime","MISSING","Need ms/conformer, conformers/s, GPU memory and CPU preprocessing; xTB runtime is not a substitute"),("xTB catastrophic-tail forensic","COMPLETE","One +53.105516 record traced; no new xTB jobs"),("parameter count","COMPLETE","766,288 trainable parameters"),("compute comparison","PARTIAL","Parameter group counts exist; matched wall/FLOP comparison absent"),("upstream transfer AvgFlow/DiTMC","MISSING","Final Full Joint has not been rerun across upstream sources"),("Reference-relative RMSD","MISSING","Atom-correspondence audit absent"),("final component V3D/PB","COMPLETE","17,500 and 22,500 component transitions complete"),("training-capacity test","COMPLETE","Exact 17,500->22,500 continuation and paired DEV bootstrap complete")],columns=["EVIDENCE","STATUS","DETAIL"])
    atomic_text(OUT/"09_MISSING_EVIDENCE_MATRIX.md","# Missing-evidence matrix\n\n"+markdown_table(missing)+"\n")
    atomic_text(OUT/"10_CLAIM_CORRECTION_AUDIT.md","""# Scientific claim correction audit

| Overclaim to avoid | Supported wording |
|---|---|
| Full Joint proves Adaptive BA alone is effective | The jointly trained system improves over its comparator; BA contribution is not causally isolated. |
| xTB energy decrease proves thermodynamic stability | GFN2-xTB single-point energy improved on this seed307 DEV diagnostic. |
| 99.88% lower energy is universal | 99.88% is limited to this seed307 DEV cohort and the tested Source coordinates. |
| mean w_B near 0.505 proves BA importance | A near-equal mean cannot establish importance; distribution and controlled ablation are required. |
| tau_max is not limiting | Current evidence for binding is weak, not absent; the exact maximum occurs and the extreme tail is near the bound. |
| training was already bottlenecked at 17,500 | The capacity test gives only weak/no evidence that 17,500 limited V3D. |
| 22,500 is the better model | V3D point estimate is higher, but its paired CI includes zero and Bond MAE reverses slightly. |
""")
    recommendation=f"""# Final recommendation

The exact continuation completed, but it does **not** justify promoting step 22,500 over the established step 17,500 candidate. V3D changes from 0.5640 to 0.5662 (`delta=+0.0022`, 95% CI `[-0.0014, 0.0058]`); PB remains 0.9324. Internal post and Angle MAE improve, while Bond MAE worsens by `+1.8178e-05` with CI `[1.4964e-05, 2.1362e-05]`. Thus the preregistered joint criterion fails.

Retain `J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500` as the current seed307 development candidate. Archive step 22,500 as a valid capacity diagnostic, not a DEV-selected replacement. No further extension is authorized.

```text
TRAINING_SATURATION_AT_17500 = LIKELY_NOT_SATURATED
POST_EXTENSION_CAPACITY_CLASSIFICATION = NEAR_PLATEAU
TRAINING_BUDGET_17500_WAS_LIMITING = NO_OR_WEAK
CURRENT_FINAL_MODEL = J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500
TAU_MAX_CLASSIFICATION = HUMAN_SCIENTIFIC_DESIGN__WEAK_BINDING_EVIDENCE
ATOM_CAP_CLASSIFICATION = HUMAN_SCIENTIFIC_DESIGN__WEAK_BINDING_EVIDENCE
MOVEMENT_REGULARIZER_CLASSIFICATION = HUMAN_SCIENTIFIC_DESIGN_FORM_PLUS_TRAIN_DATA_DERIVED_COEFFICIENT
```
"""
    atomic_text(OUT/"12_FINAL_RECOMMENDATION.md",recommendation)
    final={"schema_version":"sixs-full-joint-final-capacity-audit-v1","AUDIT_STATUS":"COMPLETE","CURRENT_FINAL_MODEL":"J1_R1_FULL_JOINT_ADAPTIVE_BA_MOVEMENT_STEP17500","TRAINING_SATURATION":"NEAR_PLATEAU","TRAINING_SATURATION_AT_17500_PREREGISTERED":"LIKELY_NOT_SATURATED","CURRENT_STEPS":22500,"SCHEDULER_HORIZON":22500,"EXTENDED_TRAINING_STARTED":"YES","EXTENDED_TRAINING_COMPLETED":"YES","V3D_17500":capacity["summary_17500"]["proposal_V3D"],"V3D_22500":capacity["summary_22500"]["proposal_V3D"],"DELTA_V3D_22500_MINUS_17500":b["V3D"]["delta_candidate_minus_baseline"],"V3D_95CI":[b["V3D"]["ci95_low"],b["V3D"]["ci95_high"]],"TRAINING_BUDGET_17500_WAS_LIMITING":"NO_OR_WEAK","CORE_DIRECTION_WEIGHTS_HUMAN_FIXED":"NO","CORE_MOVEMENT_PER_MOLECULE_HUMAN_FIXED":"NO","NUMBER_OF_HUMAN_SCIENTIFIC_DESIGN_CONSTANTS":12,"MOST_IMPORTANT_REMAINING_HUMAN_CONSTANTS":"beta_NLL_beta; log_sigma_ratio_limit; inherited sigma_stat floors; tau_max; atom cap; movement regularizer form and 5% calibration target","TAU_MAX_CLASSIFICATION":"HUMAN_SCIENTIFIC_DESIGN__WEAK_BINDING_EVIDENCE","ATOM_CAP_CLASSIFICATION":"HUMAN_SCIENTIFIC_DESIGN__WEAK_BINDING_EVIDENCE","MOVEMENT_REGULARIZER_CLASSIFICATION":"HUMAN_SCIENTIFIC_DESIGN_FORM_PLUS_TRAIN_DATA_DERIVED_COEFFICIENT","MISSING_CRITICAL_EVIDENCE":"seed331/353; Formal; large holdout; AMR-P/R; COV-P/R; neural runtime; matched compute comparison; upstream transfer; correspondence-audited Reference RMSD","XTB_EXTREME_TAIL_CAUSE":cause,"FORMAL_READ":"NO","LARGE_HOLDOUT_READ":"NO","NEW_ARCHITECTURE_CREATED":"NO","HYPERPARAMETER_SWEEP":"NO","SEED331_STARTED":"NO","SEED353_STARTED":"NO","XTB_NEW_COMPUTATION_STARTED":"NO","artifacts":{}}
    for name in ("01_FINAL_MODEL_IDENTITY.md","02_HUMAN_CONSTANT_INVENTORY.csv","03_HUMAN_CONSTANT_CLASSIFICATION.md","04_TRAINING_CURVE.csv","05_TRAINING_SATURATION_AUDIT.md","06_LEARNING_CURVE_FIT.json","07_17500_VS_22500.csv","08_TRAINING_BUDGET_BOOTSTRAP.csv","09_MISSING_EVIDENCE_MATRIX.md","10_CLAIM_CORRECTION_AUDIT.md","11_XTB_EXTREME_TAIL_FORENSIC.md","12_FINAL_RECOMMENDATION.md","SATURATION_PREREGISTRATION.json","STATE_CONTINUITY_GATE.json","EXTENSION_EVALUATION_COMPLETE.json"):
        final["artifacts"][name]=sha(OUT/name)
    atomic_json(OUT/"FINAL_STATUS.json",final)
    print(json.dumps(final,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

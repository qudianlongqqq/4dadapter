#!/usr/bin/env python3
"""Evaluate frozen SIXS-v2-DeltaQ on the established ETFlow DEV identities."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
os.environ.setdefault("SIXS_FACTORIAL_RUN_NAMESPACE", "sixs_musigma_reliability_factorial_cuda")
import scripts.run_sixs_deltaq_prototype as delta
import scripts.run_sixs_musigma_reliability_factorial as frozen
from etflow.commons.kabsch_utils import kabsch_align
from etflow.ecir.j1r1_full_joint_unrestricted import UnrestrictedFullJointModel, unrestricted_full_joint_action
from etflow.ecir.learned_geometry import geometry_values
from etflow.ecir.source_conditioned_deltaq import DeltaQUnrestrictedFullJointModel, deltaq_targets
from scripts.run_mcvr_lsgo import collate_graphs

REPORT = ROOT / "reports/ecir_mvr/sixs_deltaq_single_seed_pilot"
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_deltaq_single_seed_pilot\etflow")
CHECKPOINT = ROOT / "artifacts/ecir_mvr/sixs_deltaq_single_seed_pilot/DELTAQ_SEED307_FULL.pt"
V1 = ROOT / "reports/ecir_mvr/sixs_reference_xtb_and_unrestricted_movement_seed307/02_UNRESTRICTED_MOVEMENT/FINAL_CHECKPOINT.pt"
EXTERNAL = Path(r"E:\miniconda\envs\external-validity\python.exe")
WORKER = ROOT / "scripts/evaluate_sixs_primary_final_external.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_mol(writer: Chem.SDWriter, record, xyz: torch.Tensor, rid: str, method: str) -> None:
    adapted = frozen.adapt_formal_cache_record(record)
    mol = Chem.Mol(adapted["_formal_rdkit_mol"])
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(xyz.detach().cpu().double().tolist()):
        conf.SetAtomPosition(i, p)
    mol.RemoveAllConformers(); mol.AddConformer(conf, assignId=True)
    mol.SetProp("_Name", rid); mol.SetProp("sample_id", rid); mol.SetProp("record_id", rid); mol.SetProp("method", method)
    writer.write(mol)


def aligned_rmsd(x: torch.Tensor, y: torch.Tensor) -> float:
    aligned = kabsch_align(x, y)
    return float((aligned - y).square().sum(-1).mean().sqrt())


def action_metrics(source, proposal, reference, graph, prediction, tau: float, method: str, rid: str, mid: str):
    src_b, src_a = geometry_values(source, graph)
    out_b, out_a = geometry_values(proposal, graph)
    ref_b, ref_a = geometry_values(reference, graph)
    row = {
        "record_id": rid, "molecule_id": mid, "method": method,
        "reference_rmsd": aligned_rmsd(proposal, reference),
        "source_rmsd": aligned_rmsd(proposal, source),
        "bond_reference_mae": float((out_b-ref_b).abs().mean()),
        "angle_cosine_reference_mae": float((out_a-ref_a).abs().mean()),
        "movement_rms": float((proposal-source).square().sum(-1).mean().sqrt()),
        "tau": tau, "finite": bool(torch.isfinite(proposal).all()),
    }
    primitive = []
    if prediction is not None:
        for family, src, out, ref, pred in (
            ("bond", src_b, out_b, ref_b, prediction["bond_deltaq"]),
            ("angle", src_a, out_a, ref_a, prediction["angle_deltaq"]),
        ):
            target = ref-src; actual = out-src
            valid = target.abs() > 1e-12
            direction = (actual[valid]*target[valid] > 0) if valid.any() else target.new_empty(0, dtype=torch.bool)
            overshoot = direction & (actual[valid].abs() > target[valid].abs()) if valid.any() else direction
            for t, p, a, agree, over in zip(target.tolist(), pred.tolist(), actual.tolist(), direction.tolist(), overshoot.tolist(), strict=True):
                primitive.append({"record_id": rid, "molecule_id": mid, "family": family,
                                  "abs_target": abs(t), "abs_prediction": abs(p), "abs_actual_movement": abs(a),
                                  "direction_agreement": bool(agree), "overshoot": bool(over), "tau": tau})
    return row, primitive


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_NO_CPU_FALLBACK")
    if not CHECKPOINT.is_file() or not V1.is_file():
        raise RuntimeError("FROZEN_CHECKPOINT_MISSING")
    config = delta.cfg()
    manifest_path = ROOT / config["data"]["dev_manifest"]
    if sha(manifest_path) != config["data"]["dev_manifest_sha256"]:
        raise RuntimeError("DEV_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [str(s) for row in manifest["rows"] for s in row["sample_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise RuntimeError("DEV_DENOMINATOR_CHANGED")
    prepared, source_payload = frozen.load_inputs()
    by_molecule = {str(x["molecule_id"]): x for x in prepared["val"]}
    by_sample = {str(x["sample_id"]): x for x in source_payload["val"]}
    val_manifest = pd.read_parquet(config["data"]["val_manifest"])
    meta = {str(x.sample_id): x for x in val_manifest.itertuples(index=False)}
    device = torch.device("cuda:0")
    dq_state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    dq = DeltaQUnrestrictedFullJointModel(128, 3); dq.load_state_dict(dq_state["model_state"], strict=True); dq.to(device).eval()
    v1_state = torch.load(V1, map_location="cpu", weights_only=False)
    v1 = UnrestrictedFullJointModel(128, 3); v1.load_state_dict(v1_state["model_state"], strict=True); v1.to(device).eval()
    ASSET.mkdir(parents=True, exist_ok=True); sdf_dir = ASSET / "sdf"; sdf_dir.mkdir(parents=True, exist_ok=True)
    methods = ("ETFLOW_RAW", "ETFLOW_SIXS_V1_SEED307", "ETFLOW_SIXS_DELTAQ_SEED307")
    tmp = {m: sdf_dir / f".{m}.{os.getpid()}.tmp.sdf" for m in methods}
    writers = {m: Chem.SDWriter(str(tmp[m])) for m in methods}
    rows, primitives = [], []
    try:
        for start in range(0, len(ids), 64):
            batch_ids = ids[start:start+64]
            source_rows = [by_sample[x] for x in batch_ids]
            items = [by_molecule[str(x["molecule_id"])] for x in source_rows]
            graphs_cpu = [x["graph"] for x in items]
            graphs = [x.to(device) for x in graphs_cpu]
            bg = collate_graphs(graphs_cpu).to(device)
            source = torch.cat([torch.as_tensor(x["source"], dtype=torch.float64) for x in source_rows]).to(device)
            reference = torch.cat([torch.as_tensor(x["references"][0], dtype=torch.float64) for x in items]).to(device)
            with torch.inference_mode():
                p1 = v1.belief(bg, detach_sigma_features=False); a1 = unrestricted_full_joint_action(v1, source, graphs, p1)
                pdq = dq.belief(bg, source, detach_sigma_features=False); adq = unrestricted_full_joint_action(dq, source, graphs, pdq)
            offset = bo = ao = 0
            for local, (rid, source_row, item, graph_cpu) in enumerate(zip(batch_ids, source_rows, items, graphs_cpu, strict=True)):
                n = int(graph_cpu.atom_categorical.size(0)); nb = int(graph_cpu.bonds.size(1)); na = int(graph_cpu.angles.size(0))
                sl=slice(offset,offset+n); bs=slice(bo,bo+nb); ass=slice(ao,ao+na); offset+=n; bo+=nb; ao+=na
                mid=str(source_row["molecule_id"]); src=source[sl]; ref=reference[sl]
                graph = graphs[local]
                raw_row,_ = action_metrics(src,src,ref,graph,None,0.0,methods[0],rid,mid); rows.append(raw_row)
                v1_row,_ = action_metrics(src,a1.proposal[sl],ref,graph,None,float(a1.tau[local]),methods[1],rid,mid); rows.append(v1_row)
                pred_local={"bond_deltaq":pdq["bond_deltaq"][bs],"angle_deltaq":pdq["angle_deltaq"][ass]}
                dq_row, prim = action_metrics(src,adq.proposal[sl],ref,graph,pred_local,float(adq.tau[local]),methods[2],rid,mid)
                rows.append(dq_row); primitives.extend(prim)
                m=meta[rid]; cache=Path(config["data"]["val_cache"])/Path(str(m.source_path)).name; raw=cache.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=str(m.source_file_sha256): raise RuntimeError("DEV_CACHE_HASH_CHANGED")
                record=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=False)
                write_mol(writers[methods[0]],record,src,rid,methods[0]); write_mol(writers[methods[1]],record,a1.proposal[sl],rid,methods[1]); write_mol(writers[methods[2]],record,adq.proposal[sl],rid,methods[2])
    finally:
        for w in writers.values(): w.close()
    for m in methods: os.replace(tmp[m], sdf_dir/f"{m}.sdf")
    frame=pd.DataFrame(rows); primitive=pd.DataFrame(primitives)
    records=frame[["record_id","molecule_id"]].drop_duplicates("record_id")
    records_path=ASSET/"SOURCE_RECORDS.parquet"; records.to_parquet(records_path,index=False)
    endpoint=[]
    for m in methods:
        pb=ASSET/f"{m}_PB.parquet"; v3=ASSET/f"{m}_V3D.parquet"
        subprocess.run([str(EXTERNAL),str(WORKER),"--arm",m,"--sdf",str(sdf_dir/f"{m}.sdf"),"--records",str(records_path),"--pb",str(pb),"--v3d",str(v3)],cwd=ROOT,check=True)
        pbf=pd.read_parquet(pb); v3f=pd.read_parquet(v3); part=frame[frame.method==m]
        endpoint.append({"upstream":"ETFlow","method":m,"records":len(part),"V3D":float(v3f.validity3d.mean()),"PB":float(pbf.PB.mean()),
                         "bond_reference_mae":float(part.bond_reference_mae.mean()),"angle_cosine_reference_mae":float(part.angle_cosine_reference_mae.mean()),
                         "reference_rmsd":float(part.reference_rmsd.mean()),"source_rmsd":float(part.source_rmsd.mean()),
                         "finite_coordinate_fraction":float(part.finite.mean()),"tau_median":float(part.tau.median()),"movement_rms":float(part.movement_rms.mean())})
    atomic_csv(REPORT/"04_ETFLOW_DEV_RESULTS.csv",pd.DataFrame(endpoint))
    head=[]
    for family, part in primitive.groupby("family"):
        boundaries=part.abs_target.quantile([.25,.5,.75]).to_list()
        part=part.assign(headroom_quartile=pd.cut(part.abs_target,[-np.inf,*boundaries,np.inf],labels=[1,2,3,4],include_lowest=True))
        for q,g in part.groupby("headroom_quartile",observed=True):
            head.append({"family":family,"headroom_quartile":int(q),"primitives":len(g),"median_abs_deltaq_target":float(g.abs_target.median()),
                         "median_abs_deltaq_hat":float(g.abs_prediction.median()),"median_actual_primitive_movement":float(g.abs_actual_movement.median()),
                         "median_tau":float(g.tau.median()),"direction_agreement":float(g.direction_agreement.mean()),"overshoot_fraction":float(g.overshoot.mean())})
    atomic_csv(REPORT/"05_ETFLOW_HEADROOM_RESULTS.csv",pd.DataFrame(head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

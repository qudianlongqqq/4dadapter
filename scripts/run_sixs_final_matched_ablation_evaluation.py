#!/usr/bin/env python
"""Resumable inference and frozen external evaluation for final SIXS ablations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import torch
from torch import nn

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_primary_final_coordinates as primary
from etflow.ecir.j1r1_full_joint_unrestricted import UnrestrictedFullJointModel, unrestricted_full_joint_action
from scripts.run_sixs_final_matched_ablation_training import EqualBA, UnitReliability

PRIMARY_ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1")
EXTERNAL_PYTHON = Path(r"E:\miniconda\envs\external-validity\python.exe")
FIXED_TAU = 0.004321542759325059


class MaskedBA(nn.Module):
    def __init__(self, inner: nn.Module, keep: int):
        super().__init__(); self.inner = inner; self.keep = keep
    def forward(self, graph_embedding):
        values = self.inner(graph_embedding)
        mask = values.new_zeros(values.shape); mask[:, self.keep] = 1
        return values * mask


class FixedTau(nn.Module):
    def forward(self, graph_embedding, normalized_state):
        return graph_embedding.new_full((graph_embedding.shape[0],), FIXED_TAU)


def load_model(method, device):
    model = UnrestrictedFullJointModel(128, 3)
    variant = method["variant"]
    if variant == "reliability_off": model.reliability = UnitReliability()
    if variant == "equal_ba": model.adaptive_ba = EqualBA()
    payload = torch.load(method["checkpoint"], map_location="cpu", weights_only=False)
    # Newly trained checkpoint hashes are recorded in 02_TRAINING_INTEGRITY;
    # binding them at load time avoids mutating the pre-training protocol.
    method.setdefault("checkpoint_sha256", primary.sha256_file(method["checkpoint"]))
    if int(payload.get("step", payload.get("optimizer_steps", -1))) != 17500:
        raise RuntimeError(f"checkpoint is not step17500: {method['id']}")
    model.load_state_dict(payload["model_state"], strict=True)
    if variant == "bond_only": model.adaptive_ba = MaskedBA(model.adaptive_ba, 0)
    elif variant == "angle_only": model.adaptive_ba = MaskedBA(model.adaptive_ba, 1)
    elif variant == "fixed_tau": model.magnitude = FixedTau()
    model._sixs_ablation_variant = variant
    return model.to(device).eval()


def action(model, source, graphs, prediction):
    local = dict(prediction)
    if model._sixs_ablation_variant == "fixed_sigma_action":
        # Reliability consumes learned primitive features in model dtype.  The
        # frozen graph statistics are serialized as float64, so align them to
        # the corresponding prediction tensor before entering the action path.
        local["bond_sigma"] = torch.cat([g.bond_fixed[:, 1] for g in graphs]).to(prediction["bond_sigma"])
        local["angle_sigma"] = torch.cat([g.angle_fixed[:, 1] for g in graphs]).to(prediction["angle_sigma"])
    return unrestricted_full_joint_action(model, source, graphs, local)


def materialize(args, protocol, chunks):
    topology = {}
    for path in chunks:
        for item in torch.load(path, map_location="cpu", weights_only=False)["items"]:
            topology[int(item["final_molecule_index"])] = {"mol_binary": item["mol_binary"]}
    for method in protocol["model_methods"]:
        target = args.output_dir / "methods" / method["id"]
        ready = target / "COORDINATES_READY.json"
        if ready.is_file() and (target / "PER_RECORD.parquet").is_file() and (target / "COORDINATES.sdf").is_file():
            continue
        rows = []
        for path in chunks:
            chunk = target / "chunks" / path.name
            if not chunk.is_file(): raise RuntimeError(f"missing inference chunk: {chunk}")
            rows.extend(torch.load(chunk, map_location="cpu", weights_only=False))
        rows.sort(key=lambda row: (row["final_molecule_index"], row["etflow_record_index"]))
        if len(rows) != 5000 or len({row["record_id"] for row in rows}) != 5000:
            raise RuntimeError(f"record alignment failure: {method['id']}")
        primary.write_sdf(target / "COORDINATES.sdf", rows, topology, method["id"])
        scalars = [{k: v for k, v in row.items() if k not in {"proposal_coordinates", "bond_sigma", "angle_sigma", "bond_reliability", "angle_reliability"}} for row in rows]
        primary.atomic_parquet(target / "PER_RECORD.parquet", pd.DataFrame(scalars))
        primary.atomic_json(ready, {"status":"PASS", "method":method["id"], "molecules":2500, "records":5000,
            "sdf_sha256":primary.sha256_file(target / "COORDINATES.sdf"), "per_record_sha256":primary.sha256_file(target / "PER_RECORD.parquet"),
            "protocol_sha256":primary.sha256_file(args.protocol), "model_training_performed":False})


def coordinates(args, protocol):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable for neural inference")
    cache = PRIMARY_ASSET / "topology_reference_cache" / "COMPLETE.json"
    cache_value = json.loads(cache.read_text(encoding="utf-8"))
    chunks = [Path(value) for value in cache_value["chunks"]]
    if not all(path.is_file() for path in chunks): raise RuntimeError("frozen primary topology cache incomplete")
    primary.load_model = load_model
    primary.unrestricted_full_joint_action = action
    for method in protocol["model_methods"]:
        target = args.output_dir / "methods" / method["id"] / "COORDINATES_READY.json"
        if target.is_file(): continue
        primary.infer_method(args, protocol, method, chunks, smoke=False)
    materialize(args, protocol, chunks)


def external(args, protocol):
    worker = ROOT / "scripts/evaluate_sixs_primary_final_external.py"
    for method in protocol["model_methods"]:
        target = args.output_dir / "methods" / method["id"]
        pb, v3d = target / "POSEBUSTERS.parquet", target / "VALIDITY3D.parquet"
        if pb.is_file() and v3d.is_file(): continue
        logs = args.report_dir / "stage_logs"; logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"external_{method['id']}.stdout.log").open("a", encoding="utf-8") as out, (logs / f"external_{method['id']}.stderr.log").open("a", encoding="utf-8") as err:
            result = subprocess.run([str(EXTERNAL_PYTHON), str(worker), "--arm", method["id"], "--sdf", str(target/"COORDINATES.sdf"),
                "--records", str(target/"PER_RECORD.parquet"), "--pb", str(pb), "--v3d", str(v3d)], cwd=ROOT, stdout=out, stderr=err)
        if result.returncode != 0: raise RuntimeError(f"external evaluation failed: {method['id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("coordinates", "external"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    for key in ("protocol","output_dir","report_dir"): setattr(args,key,getattr(args,key).resolve())
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True); args.report_dir.mkdir(parents=True, exist_ok=True)
    args.primary = Path(protocol["bindings"]["primary_manifest"])
    args.source_manifest = Path(protocol["bindings"]["source_record_manifest"])
    args.source_asset_freeze = Path(protocol["bindings"]["source_asset_freeze"])
    args.smoke_molecules = 3
    if args.stage == "coordinates": coordinates(args, protocol)
    else: external(args, protocol)
    return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python
"""Strictly label-free rollout for Gated Molecular Kinematic Flow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.provenance import collect_run_provenance
from etflow.commons.run_state import update_run_state
from etflow.data.flexbond_eval_manifest import build_manifest_aware_sample_payload,load_eval_manifest,limit_manifest_molecules,validate_dataset_against_manifest
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset
from etflow.models.gated_kinematic_flow import GatedKinematicFlowLightningModule


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True); parser.add_argument("--config",required=True)
    parser.add_argument("--cache_dir",required=True); parser.add_argument("--manifest",required=True)
    parser.add_argument("--split",default="test"); parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--refinement_steps",type=int,default=10,choices=(1,5,10,20))
    parser.add_argument("--update_scale",type=float,default=.5); parser.add_argument("--max_displacement",type=float)
    parser.add_argument("--max_coordinate_norm",type=float,default=1000.0); parser.add_argument("--max_molecules",type=int)
    parser.add_argument("--device",default="cpu"); parser.add_argument("--gate_override",choices=("none","all_one","all_zero"),default="none")
    parser.add_argument("--torsion_rate_scale_override",type=float); parser.add_argument("--disable_orthogonalization",action="store_true")
    parser.add_argument("--save_trajectory_metrics",action="store_true")
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True,exist_ok=True); run_dir=args.output.parent
    update_run_state(run_dir,"started",stage="sampling",expected_outputs=[args.output.name])
    try:
        model=GatedKinematicFlowLightningModule.load_from_checkpoint(args.checkpoint,map_location=args.device).to(args.device).eval()
        dataset=FlexBondInferenceDataset(args.cache_dir,args.split); manifest=load_eval_manifest(args.manifest)
        selected_manifest=manifest
        if args.max_molecules is not None: selected_manifest=limit_manifest_molecules(manifest,args.max_molecules)
        by_id=validate_dataset_against_manifest(dataset,selected_manifest); records=[]; trajectory=[]
        for manifest_row in selected_manifest["records"]:
            data=by_id[str(manifest_row["sample_id"])].to(args.device)
            refined,diagnostics=model.refine(data,args.refinement_steps,args.update_scale,args.max_displacement,
                args.max_coordinate_norm,args.gate_override,args.torsion_rate_scale_override,
                args.disable_orthogonalization,args.save_trajectory_metrics)
            stable=diagnostics["stable"]
            records.append({"mol_id":data.mol_id,"sample_id":data.sample_id,"source_mol_id":data.source_mol_id,
                "smiles":data.smiles,"atomic_numbers":data.atomic_numbers.cpu(),"x_init":data.x_init.cpu(),
                "x_init_hash":str(manifest_row["x_init_hash"]),"x_refined":refined.cpu() if stable else None,
                "num_rotatable_bonds":int(data.num_rotatable_bonds.item()),"method_name":"gated_kinematic_adapter",
                "motion_mode":model.motion_mode,"status":"success" if stable else "failed",
                "checkpoint_path":str(Path(args.checkpoint).resolve()),"config_path":str(Path(args.config).resolve()),
                "refinement_steps":args.refinement_steps,"update_scale":args.update_scale,"alpha":args.update_scale,
                "max_displacement":args.max_displacement,"gate_override":args.gate_override,
                "torsion_rate_scale_override":args.torsion_rate_scale_override,
                "disable_orthogonalization":args.disable_orthogonalization,**{k:v for k,v in diagnostics.items() if k!="trajectory"}})
            for row in diagnostics["trajectory"]: trajectory.append({"sample_id":str(data.sample_id),**row})
        provenance=collect_run_provenance(config_path=args.config,checkpoint_path=args.checkpoint,cache_path=args.cache_dir)
        provenance.update({"label_free":True,"gate_override":args.gate_override,
            "torsion_rate_scale_override":args.torsion_rate_scale_override,
            "disable_orthogonalization":args.disable_orthogonalization})
        failures=sum(r["status"]!="success" for r in records)
        payload=build_manifest_aware_sample_payload(records=records,manifest=manifest,
            manifest_path=args.manifest,selected_manifest=selected_manifest,split=args.split,
            inference_cache_path=args.cache_dir,inference_by_id=by_id,
            extra={"provenance":provenance,"failure_count":failures,
                "failure_rate":failures/len(records) if records else 0.0})
        torch.save(payload,args.output)
        if args.save_trajectory_metrics:
            path=args.output.with_name(args.output.stem+"_trajectory.csv")
            with path.open("x",newline="",encoding="utf-8-sig") as handle:
                fields=list(trajectory[0]) if trajectory else ["sample_id","rollout_step"]
                writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(trajectory)
        update_run_state(run_dir,"completed",stage="sampling",output=str(args.output),num_records=len(records))
    except KeyboardInterrupt:
        (run_dir/"STOPPED_REASON.txt").write_text("KeyboardInterrupt\n",encoding="utf-8")
        update_run_state(run_dir,"stopped",stage="sampling",reason="KeyboardInterrupt"); raise
    except Exception as exc:
        update_run_state(run_dir,"failed",stage="sampling",error=repr(exc)); raise


if __name__=="__main__": main()

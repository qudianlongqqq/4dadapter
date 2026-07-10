#!/usr/bin/env python
"""Synthetic parameter/timing comparison; never reads formal experiment data."""

from __future__ import annotations

import argparse,json,time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
bootstrap()

import torch
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule
from etflow.models.gated_kinematic_flow import GatedKinematicFlowLightningModule


def count(model):return sum(parameter.numel() for parameter in model.parameters())


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cpu");parser.add_argument("--repeats",type=int,default=20);parser.add_argument("--output",type=Path)
    args=parser.parse_args();device=torch.device(args.device);atoms=32
    edge=torch.stack([torch.arange(atoms-1),torch.arange(1,atoms)]);edge=torch.cat([edge,edge.flip(0)],dim=1)
    batch={"x_init":torch.randn(atoms,3,device=device),"node_attr":torch.randn(atoms,10,device=device),
        "edge_index":edge.to(device),"edge_attr":torch.zeros(edge.size(1),1,device=device),
        "rotatable_bond_index":torch.tensor([[4,9,14,19],[5,10,15,20]],device=device),
        "atom_bond_influence_index":torch.empty((2,0),dtype=torch.long,device=device),
        "batch":torch.zeros(atoms,dtype=torch.long,device=device)}
    legacy=FlexBondOptimizerLightningModule().to(device).eval();gated=GatedKinematicFlowLightningModule().to(device).eval()
    def timing(model):
        for _ in range(2):model(batch)
        if device.type=="cuda":torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(args.repeats):model(batch)
        if device.type=="cuda":torch.cuda.synchronize()
        return (time.perf_counter()-start)/args.repeats
    result={"legacy_parameter_count":count(legacy),"gated_parameter_count":count(gated),
        "legacy_head_scalars_per_joint":4,"gated_head_scalars_per_joint":2,
        "legacy_forward_seconds":timing(legacy),"gated_forward_seconds":timing(gated),
        "asymptotic_jacobian_apply":"O(sum affected atoms over joints)",
        "dense_training_projection":"O(3*N*M + min(3*N,M)*M^2)",
        "inference_projection":"matrix-free CG; O(iterations * sum affected atoms)"}
    text=json.dumps(result,indent=2);print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+"\n",encoding="utf-8")


if __name__=="__main__":main()

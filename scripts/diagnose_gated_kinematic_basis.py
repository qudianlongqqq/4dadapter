#!/usr/bin/env python
"""Offline oracle comparison of torsion, independent-4D, and global-4D bases."""

from __future__ import annotations

import argparse,csv,json,math,random
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import torch

from etflow.commons.flexbond_jacobian import apply_bond_jacobian,build_bond_local_frame,identify_target_bonds,solve_q_star_least_squares
from etflow.commons.jacobian_4d_velocity import build_atom_jacobian
from etflow.commons.kinematic_projection import decompose_target
from etflow.commons.molecular_kinematics import build_molecular_kinematic_topology
from etflow.commons.provenance import collect_run_provenance
from etflow.commons.run_state import atomic_write_json,update_run_state
from etflow.commons.torsion_kinematic_jacobian import build_dense_jacobian
from etflow.data.flexbond_optimizer_dataset import FlexBondOptimizerDataset


FIELDS=("sample_id","molecule_id","t","num_atoms","num_rotatable_bonds","valid_joint_count",
    "effective_rank","target_norm","torsion_projection_norm","torsion_residual_norm",
    "torsion_explained_ratio","global_torsion_reconstruction_error",
    "independent_4d_reconstruction_error","global_4d_reconstruction_error","orthogonality_error",
    "condition_estimate","topology_status","rank_deficient","no_valid_joint","finite")


def rms(value): return float(value.square().mean().sqrt()) if value.numel() else 0.0


def global_4d_basis(pos,targets):
    bonds=targets["anchor_index"].numel()
    if not bonds:return pos.new_zeros((pos.numel(),0))
    frame,valid=build_bond_local_frame(pos,targets["anchor_index"],targets["moving_index"],
        targets["affected_atom_index"],targets["affected_bond_index"])
    atoms,joints=targets["affected_atom_index"],targets["affected_bond_index"]
    lever=pos[atoms]-pos[targets["anchor_index"][joints]]
    local=build_atom_jacobian(frame[joints,:,0],frame[joints],lever)*valid[joints,None,None]
    dense=pos.new_zeros((pos.size(0),3,4*bonds))
    for contribution,(atom,joint) in zip(local,zip(atoms.tolist(),joints.tolist())):
        dense[atom,:,4*joint:4*joint+4]+=contribution
    return dense.reshape(pos.numel(),4*bonds)


@torch.no_grad()
def analyze(data,t_value):
    pos=(1-t_value)*data.x_init+t_value*data.x_ref_aligned; target=data.x_ref_aligned-data.x_init
    topology=build_molecular_kinematic_topology(pos.size(0),data.edge_index,data.rotatable_bond_index).to(pos.device)
    jacobian,valid=build_dense_jacobian(pos,topology); decomposition=decompose_target(jacobian,target)
    projected=decomposition.u_kin_star; residual=decomposition.u_res_star
    target_error=max(float(target.square().sum()),1e-24)
    explained=1-float(residual.square().sum())/target_error
    torsion_recon=rms(target-(projected+residual))
    atom_batch=torch.zeros(pos.size(0),dtype=torch.long,device=pos.device)
    targets=identify_target_bonds(data.rotatable_bond_index,data.atom_bond_influence_index,atom_batch)
    q_ind,_,_=solve_q_star_least_squares(pos,target,targets,max_condition=1e12,max_q_norm=1e12)
    independent,_=apply_bond_jacobian(pos,q_ind,targets)
    global_basis=global_4d_basis(pos,targets)
    global_projection=(global_basis@torch.linalg.pinv(global_basis)@target.reshape(-1)).reshape_as(target) if global_basis.size(1) else torch.zeros_like(target)
    singular=decomposition.singular_values
    condition=float(singular[0]/singular[decomposition.rank-1]) if decomposition.rank and singular[decomposition.rank-1]>0 else float("inf")
    values={"sample_id":str(getattr(data,"sample_id",data.mol_id)),"molecule_id":str(data.mol_id),"t":t_value,
        "num_atoms":pos.size(0),"num_rotatable_bonds":int(data.num_rotatable_bonds.item()),
        "valid_joint_count":int(valid.sum()),"effective_rank":decomposition.rank,"target_norm":rms(target),
        "torsion_projection_norm":rms(projected),"torsion_residual_norm":rms(residual),
        "torsion_explained_ratio":explained,"global_torsion_reconstruction_error":torsion_recon,
        "independent_4d_reconstruction_error":rms(target-independent),
        "global_4d_reconstruction_error":rms(target-global_projection),
        "orthogonality_error":abs(float((projected.reshape(-1)*residual.reshape(-1)).sum())),
        "condition_estimate":condition,"topology_status":topology.status,
        "rank_deficient":decomposition.rank<topology.num_joints,"no_valid_joint":topology.num_joints==0}
    values["finite"]=all(math.isfinite(float(v)) for k,v in values.items() if isinstance(v,(int,float)) and k!="condition_estimate")
    return values


def summarize(rows,group):
    numeric=[name for name in FIELDS if name not in {"sample_id","molecule_id","topology_status","finite","rank_deficient","no_valid_joint"}]
    result={"group":group,"sample_count":len(rows)}
    for name in numeric:
        values=[float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        result[f"mean_{name}"]=float(np.mean(values)) if values else ""
    truth=lambda value:value if isinstance(value,bool) else str(value).strip().lower() in {"1","true","yes"}
    result["finite_fraction"]=float(np.mean([truth(row["finite"]) for row in rows])) if rows else 0.0
    return result


def write(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache_dir",required=True);parser.add_argument("--split",default="val")
    parser.add_argument("--output_dir",required=True,type=Path);parser.add_argument("--max_samples",type=int,default=500)
    parser.add_argument("--fixed_times",nargs="+",type=float,default=[.05,.1,.25,.5]);parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu");parser.add_argument("--resume",action="store_true");parser.add_argument("--skip_existing",action="store_true")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True); sample_path=args.output_dir/"sample_metrics.csv"
    if sample_path.exists() and not (args.resume or args.skip_existing): raise FileExistsError("Use --resume or a new output directory")
    existing=[];done=set()
    if sample_path.exists():
        with sample_path.open(encoding="utf-8-sig") as handle: existing=list(csv.DictReader(handle))
        done={(r["sample_id"],float(r["t"])) for r in existing}
    update_run_state(args.output_dir,"started",stage="basis_diagnostic")
    try:
        dataset=FlexBondOptimizerDataset(args.cache_dir,args.split);indices=list(range(len(dataset)));random.Random(args.seed).shuffle(indices)
        rows=existing
        for index in indices[:args.max_samples]:
            data=dataset[index].to(args.device);sample_id=str(getattr(data,"sample_id",data.mol_id))
            for t in args.fixed_times:
                if (sample_id,t) in done: continue
                rows.append(analyze(data,t));write(sample_path,rows,FIELDS)
                atomic_write_json(args.output_dir/"partial_progress.json",{"completed_rows":len(rows),"last_sample_id":sample_id,"last_t":t})
        molecule=[]
        for molecule_id in sorted({r["molecule_id"] for r in rows}): molecule.append(summarize([r for r in rows if r["molecule_id"]==molecule_id],molecule_id))
        write(args.output_dir/"molecule_metrics.csv",molecule,list(molecule[0]) if molecule else ["group","sample_count"])
        summary=[summarize(rows,"all")];fields=list(summary[0]);write(args.output_dir/"summary.csv",summary,fields)
        groups={"summary_by_time.csv":sorted({str(r["t"]) for r in rows}),
            "summary_by_rotatable_count.csv":["0-2","3-4","5","6+"],
            "summary_by_rank.csv":sorted({str(r["effective_rank"]) for r in rows}),
            "summary_by_topology_status.csv":sorted({r["topology_status"] for r in rows})}
        for filename,labels in groups.items():
            grouped=[]
            for label in labels:
                if filename=="summary_by_time.csv": chosen=[r for r in rows if str(r["t"])==label]
                elif filename=="summary_by_rank.csv": chosen=[r for r in rows if str(r["effective_rank"])==label]
                elif filename=="summary_by_topology_status.csv": chosen=[r for r in rows if r["topology_status"]==label]
                else:
                    pred={"0-2":lambda n:n<3,"3-4":lambda n:3<=n<5,"5":lambda n:n==5,"6+":lambda n:n>=6}[label]
                    chosen=[r for r in rows if pred(int(r["num_rotatable_bonds"]))]
                grouped.append(summarize(chosen,label))
            write(args.output_dir/filename,grouped,fields)
        provenance=collect_run_provenance(cache_path=args.cache_dir);provenance.update({"seed":args.seed,"fixed_times":args.fixed_times,"split":args.split})
        with (args.output_dir/"provenance.json").open("w",encoding="utf-8") as handle:json.dump(provenance,handle,indent=2)
        update_run_state(args.output_dir,"completed",stage="basis_diagnostic",completed_rows=len(rows))
    except KeyboardInterrupt:
        (args.output_dir/"STOPPED_REASON.txt").write_text("KeyboardInterrupt\n",encoding="utf-8");update_run_state(args.output_dir,"stopped",stage="basis_diagnostic");raise
    except Exception as exc:update_run_state(args.output_dir,"failed",stage="basis_diagnostic",error=repr(exc));raise


if __name__=="__main__":main()

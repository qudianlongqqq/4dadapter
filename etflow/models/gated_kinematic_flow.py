"""Gated Molecular Kinematic Flow with a Cartesian orthogonal residual."""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from etflow.commons.kinematic_projection import decompose_target, soft_gate_target
from etflow.commons.molecular_kinematics import build_molecular_kinematic_topology
from etflow.commons.refinement_utils import clip_atom_displacement
from etflow.commons.torsion_kinematic_jacobian import (
    apply_jacobian, build_dense_jacobian, project_matrix_free,
)
from etflow.models.components.light_egnn_refiner import (
    LightEGNNLayer, SinusoidalTimeEmbedding, _mlp,
)


MOTION_MODES = ("gated_global_torsion_kinematic", "legacy_flexbond4d", "cartesian")


def _field(batch: Any, name: str):
    return batch[name] if isinstance(batch, Mapping) else getattr(batch, name)


class GatedKinematicBackbone(nn.Module):
    """Shared EGNN trunk with Cartesian vector and invariant 2-scalar joint heads."""

    def __init__(self, atom_feature_dim=10, edge_attr_dim=1, hidden_dim=128,
                 edge_hidden_dim=128, time_embedding_dim=64, num_layers=6,
                 dropout=0.0, cutoff=10.0):
        super().__init__(); self.cutoff=float(cutoff); self.edge_attr_dim=int(edge_attr_dim)
        self.atom_embedding=nn.Linear(atom_feature_dim, hidden_dim)
        self.time_embedding=SinusoidalTimeEmbedding(time_embedding_dim)
        self.layers=nn.ModuleList([LightEGNNLayer(hidden_dim, edge_hidden_dim,
            edge_attr_dim, time_embedding_dim, dropout) for _ in range(num_layers)])
        self.cartesian_layer_weights=nn.Parameter(torch.zeros(num_layers))
        # parent/child atoms + parent/child fragment pools + distance + time
        self.motion_head=_mlp(4*hidden_dim+1+time_embedding_dim, edge_hidden_dim, 2, dropout)

    def encode(self, node_attr, pos, edge_index, edge_attr, atom_time):
        if edge_attr is None:
            edge_attr=pos.new_zeros((edge_index.size(1), self.edge_attr_dim))
        if edge_attr.ndim == 1: edge_attr=edge_attr[:,None]
        edge_attr=edge_attr.to(dtype=pos.dtype); time_emb=self.time_embedding(atom_time)
        h=self.atom_embedding(node_attr.to(dtype=pos.dtype)); vectors=[]
        for layer in self.layers:
            h, vector=layer(h,pos,edge_index,edge_attr,time_emb,self.cutoff); vectors.append(vector)
        weights=torch.softmax(self.cartesian_layer_weights,dim=0)
        return h, sum(w*v for w,v in zip(weights,vectors)), time_emb


class GatedKinematicFlowLightningModule(LightningModule):
    def __init__(
        self, motion_mode="gated_global_torsion_kinematic",
        atom_feature_dim=10, edge_attr_dim=1, hidden_dim=128, edge_hidden_dim=128,
        time_embedding_dim=64, num_layers=6, dropout=0.0, cutoff=10.0,
        orthogonalize_cartesian=True, use_motion_gate=True, torsion_rate_scale=1.0,
        gate_active_threshold=0.5, projection_rank_tol=1e-6,
        rate_target_ridge=1e-4, gate_rate_threshold=0.05, gate_temperature=0.02,
        gate_target_method="sigmoid_threshold", projection_cg_iterations=24,
        final_weight=1.0, kinematic_weight=1.0, residual_weight=1.0,
        rate_weight=0.1, gate_supervision_weight=0.05, gate_sparse_weight=0.001,
        gate_binary_weight=0.001, rate_reg_weight=0.0001,
        lr=2e-4, weight_decay=1e-6, grad_clip=1.0, t_min=0.0, t_max=0.25,
    ):
        super().__init__()
        if motion_mode != "gated_global_torsion_kinematic":
            raise ValueError("This checkpoint class only accepts gated_global_torsion_kinematic; legacy modes use FlexBondOptimizerLightningModule")
        if torsion_rate_scale <= 0 or gate_temperature <= 0: raise ValueError("rate scale and gate temperature must be positive")
        self.save_hyperparameters(); self.motion_mode=motion_mode
        self.backbone=GatedKinematicBackbone(atom_feature_dim,edge_attr_dim,hidden_dim,
            edge_hidden_dim,time_embedding_dim,num_layers,dropout,cutoff)
        self._last_topology_build_count=0

    def _atom_batch(self,batch,pos):
        value=batch.get("batch") if isinstance(batch,Mapping) else getattr(batch,"batch",None)
        return value if value is not None else torch.zeros(pos.size(0),dtype=torch.long,device=pos.device)

    def _topologies(self,batch,atom_batch):
        edge=_field(batch,"edge_index"); rot=_field(batch,"rotatable_bond_index")
        result=[]
        for graph in range(int(atom_batch.max())+1 if atom_batch.numel() else 1):
            atoms=torch.nonzero(atom_batch==graph,as_tuple=False).reshape(-1); start=int(atoms.min())
            edge_mask=(atom_batch[edge[0]]==graph)&(atom_batch[edge[1]]==graph)
            rot_mask=(atom_batch[rot[0]]==graph) if rot.numel() else torch.zeros(0,dtype=torch.bool,device=edge.device)
            topology=build_molecular_kinematic_topology(
                atoms.numel(), edge[:,edge_mask]-start, rot[:,rot_mask]-start)
            result.append((start,atoms.numel(),topology.to(edge.device)))
        self._last_topology_build_count += 1
        return result

    def forward(self,batch,pos:Optional[Tensor]=None,t:Optional[Tensor]=None,
                gate_override="none",torsion_rate_scale_override=None,
                disable_orthogonalization=False):
        pos=_field(batch,"x_init") if pos is None else pos
        atom_batch=self._atom_batch(batch,pos); graphs=int(atom_batch.max())+1 if atom_batch.numel() else 1
        if t is None: t=pos.new_zeros(graphs)
        t=torch.as_tensor(t,device=pos.device,dtype=pos.dtype).reshape(-1)
        if t.numel()==1 and graphs>1: t=t.expand(graphs)
        atom_time=t[atom_batch]
        start_time=time.perf_counter()
        h,v_cart_raw,time_emb=self.backbone.encode(_field(batch,"node_attr"),pos,
            _field(batch,"edge_index"),getattr(batch,"edge_attr",None) if not isinstance(batch,Mapping) else batch.get("edge_attr"),atom_time)
        backbone_time=time.perf_counter()-start_time
        topology_start=time.perf_counter(); topologies=self._topologies(batch,atom_batch)
        topology_time=time.perf_counter()-topology_start
        v_projection=torch.zeros_like(pos); v_kin=torch.zeros_like(pos)
        logits=[]; raw_rates=[]; gates=[]; bounded=[]; effective=[]; statuses=[]; valid_counts=[]
        head_time=0.0; jacobian_time=0.0; projection_time=0.0
        scale=float(self.hparams.torsion_rate_scale if torsion_rate_scale_override is None else torsion_rate_scale_override)
        if scale <= 0: raise ValueError("torsion_rate_scale_override must be positive")
        for start,count,topology in topologies:
            local_pos=pos[start:start+count]; local_h=h[start:start+count]; local_time=time_emb[start:start+count]
            statuses.append(topology.status); valid_counts.append(topology.num_joints)
            if topology.num_joints==0:
                continue
            head_start=time.perf_counter()
            fragment_pool=[]
            for fragment in topology.fragments:
                fragment_pool.append(local_h[list(fragment)].mean(dim=0))
            pools=torch.stack(fragment_pool)
            distance=(local_pos[topology.child_atom]-local_pos[topology.parent_atom]).square().sum(-1,keepdim=True)
            feature=torch.cat([local_h[topology.parent_atom],local_h[topology.child_atom],
                pools[topology.parent_fragment],pools[topology.child_fragment],distance,
                local_time[topology.parent_atom]],dim=-1)
            output=self.backbone.motion_head(feature); logit,raw=output[:,0],output[:,1]
            gate=torch.sigmoid(logit) if self.hparams.use_motion_gate else torch.ones_like(logit)
            if gate_override=="all_one": gate=torch.ones_like(gate)
            elif gate_override=="all_zero": gate=torch.zeros_like(gate)
            elif gate_override not in ("none",None): raise ValueError("gate_override must be none, all_one, or all_zero")
            bound=scale*torch.tanh(raw); rate=gate*bound
            head_time += time.perf_counter()-head_start
            jac_start=time.perf_counter(); local_vkin,geometry_valid=apply_jacobian(local_pos,rate,topology)
            jacobian_time += time.perf_counter()-jac_start
            valid_counts[-1]=int(geometry_valid.sum()); v_kin[start:start+count]=local_vkin
            if self.hparams.orthogonalize_cartesian and not disable_orthogonalization:
                project_start=time.perf_counter(); projected,_=project_matrix_free(
                    local_pos,v_cart_raw[start:start+count],topology,
                    iterations=self.hparams.projection_cg_iterations)
                projection_time += time.perf_counter()-project_start
                v_projection[start:start+count]=projected
            logits.append(logit); raw_rates.append(raw); gates.append(gate); bounded.append(bound); effective.append(rate)
        cat=lambda values: torch.cat(values) if values else pos.new_empty((0,))
        v_residual=v_cart_raw-v_projection; v_final=v_residual+v_kin
        gate=cat(gates); active=(gate>=self.hparams.gate_active_threshold).float().mean() if gate.numel() else pos.new_zeros(())
        return {"v_cart_raw":v_cart_raw,"v_cart_projection":v_projection,
            "v_residual":v_residual,"gate_logit":cat(logits),"gate":gate,
            "raw_torsion_rate":cat(raw_rates),"bounded_torsion_rate":cat(bounded),
            "effective_torsion_rate":cat(effective),"v_kin":v_kin,"v_final":v_final,
            "valid_joint_count":pos.new_tensor(valid_counts,dtype=torch.long),
            "kinematic_rank":pos.new_full((graphs,),-1,dtype=torch.long),
            "topology_status":statuses,"gate_active_fraction":active,
            "timing":{"backbone_time":backbone_time,"bond_head_time":head_time,
                "topology_time":topology_time,"jacobian_apply_time":jacobian_time,
                "projection_time":projection_time,
                "peak_gpu_memory":torch.cuda.max_memory_allocated(pos.device) if pos.is_cuda else 0,
                "num_atoms":pos.size(0),"num_joints":gate.numel()}}

    def _shared_step(self,batch,stage):
        x_init=_field(batch,"x_init"); x_ref=_field(batch,"x_ref_aligned")
        atom_batch=self._atom_batch(batch,x_init); graphs=int(atom_batch.max())+1 if atom_batch.numel() else 1
        t=x_init.new_empty(graphs).uniform_(self.hparams.t_min,self.hparams.t_max)
        x_t=(1-t[atom_batch,None])*x_init+t[atom_batch,None]*x_ref; target=x_ref-x_init
        output=self(batch,x_t,t); target_kin=torch.zeros_like(target); target_res=target.clone()
        rate_targets=[]; gate_targets=[]; ranks=[]
        topologies=self._topologies(batch,atom_batch)
        for graph,(start,count,topology) in enumerate(topologies):
            if topology.num_joints==0: ranks.append(0); continue
            jacobian,valid=build_dense_jacobian(x_t[start:start+count],topology)
            decomposition=decompose_target(jacobian,target[start:start+count],
                rank_tol=self.hparams.projection_rank_tol,
                rate_target_ridge=self.hparams.rate_target_ridge)
            target_kin[start:start+count]=decomposition.u_kin_star.detach()
            target_res[start:start+count]=decomposition.u_res_star.detach()
            rate_targets.append(decomposition.rate_star_damped.detach())
            gate_targets.append(soft_gate_target(decomposition.rate_star_damped.detach(),
                threshold=self.hparams.gate_rate_threshold,temperature=self.hparams.gate_temperature,
                method=self.hparams.gate_target_method)); ranks.append(decomposition.rank)
        output["kinematic_rank"]=x_init.new_tensor(ranks,dtype=torch.long)
        zero=output["v_final"].new_zeros(())
        loss_final=F.mse_loss(output["v_final"],target); loss_kin=F.mse_loss(output["v_kin"],target_kin)
        loss_res=F.mse_loss(output["v_residual"],target_res)
        rate_target=torch.cat(rate_targets) if rate_targets else output["effective_torsion_rate"].new_empty(0)
        gate_target=torch.cat(gate_targets) if gate_targets else output["gate"].new_empty(0)
        loss_rate=F.smooth_l1_loss(output["effective_torsion_rate"],rate_target) if rate_target.numel() else zero
        loss_gate=F.binary_cross_entropy(output["gate"],gate_target) if gate_target.numel() else zero
        gate_sparse=output["gate"].mean() if output["gate"].numel() else zero
        gate_binary=(output["gate"]*(1-output["gate"])).mean() if output["gate"].numel() else zero
        rate_reg=output["bounded_torsion_rate"].square().mean() if output["bounded_torsion_rate"].numel() else zero
        total=(self.hparams.final_weight*loss_final+self.hparams.kinematic_weight*loss_kin+
            self.hparams.residual_weight*loss_res+self.hparams.rate_weight*loss_rate+
            self.hparams.gate_supervision_weight*loss_gate+self.hparams.gate_sparse_weight*gate_sparse+
            self.hparams.gate_binary_weight*gate_binary+self.hparams.rate_reg_weight*rate_reg)
        gate=output["gate"]
        metrics={f"{stage}/loss":total,f"{stage}/final_loss":loss_final,
            f"{stage}/kinematic_loss":loss_kin,f"{stage}/residual_loss":loss_res,
            f"{stage}/rate_loss":loss_rate,f"{stage}/gate_supervision_loss":loss_gate,
            f"{stage}/gate_sparse_loss":gate_sparse,f"{stage}/gate_binary_loss":gate_binary,
            f"{stage}/rate_reg_loss":rate_reg,f"{stage}/gate_mean":gate.mean() if gate.numel() else zero,
            f"{stage}/gate_median":gate.median() if gate.numel() else zero,
            f"{stage}/active_gate_fraction":output["gate_active_fraction"],
            f"{stage}/gate_near_zero_fraction":(gate<.1).float().mean() if gate.numel() else zero,
            f"{stage}/gate_near_one_fraction":(gate>.9).float().mean() if gate.numel() else zero,
            f"{stage}/effective_rate_norm":output["effective_torsion_rate"].square().mean().sqrt() if gate.numel() else zero,
            f"{stage}/bounded_rate_norm":output["bounded_torsion_rate"].square().mean().sqrt() if gate.numel() else zero}
        for name,value in output["timing"].items():
            metrics[f"{stage}/performance/{name}"]=total.new_tensor(float(value))
        self.log_dict(metrics,on_step=stage=="train",on_epoch=True,batch_size=graphs,sync_dist=True)
        return total

    def training_step(self,batch,batch_idx): return self._shared_step(batch,"train")
    def validation_step(self,batch,batch_idx): return self._shared_step(batch,"val")
    def configure_optimizers(self): return torch.optim.AdamW(self.parameters(),lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
    def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_norm_(self.parameters(),self.hparams.grad_clip)

    @torch.no_grad()
    def refine(self,batch,refinement_steps=10,update_scale=1.0,max_displacement=None,
               max_coordinate_norm=1000.0,gate_override="none",
               torsion_rate_scale_override=None,disable_orthogonalization=False,
               save_trajectory_metrics=False):
        x=_field(batch,"x_init").clone(); trajectory=[]; stable=True; reason=""
        gate_means=[]; active_fractions=[]; rate_norms=[]; timing_rows=[]
        for step in range(refinement_steps):
            t=x.new_tensor(step/max(refinement_steps-1,1)); output=self(batch,x,t,
                gate_override=gate_override,torsion_rate_scale_override=torsion_rate_scale_override,
                disable_orthogonalization=disable_orthogonalization)
            raw=float(update_scale)/refinement_steps*output["v_final"]
            gate_means.append(float(output["gate"].mean()) if output["gate"].numel() else 0.0)
            active_fractions.append(float(output["gate_active_fraction"]))
            rate_norms.append(float(output["effective_torsion_rate"].square().mean().sqrt()) if output["gate"].numel() else 0.0)
            timing_rows.append(output["timing"])
            update,clipped=clip_atom_displacement(raw,max_displacement=max_displacement)
            candidate=x+update; finite=bool(torch.isfinite(candidate).all())
            bounded=finite and bool(torch.linalg.norm(candidate,dim=-1).max()<max_coordinate_norm)
            if save_trajectory_metrics:
                trajectory.append({"rollout_step":step,"update_norm":float(torch.linalg.norm(update,dim=-1).mean()),
                    "cartesian_norm":float(torch.linalg.norm(output["v_residual"],dim=-1).mean()),
                    "kinematic_norm":float(torch.linalg.norm(output["v_kin"],dim=-1).mean()),
                    "gate_mean":float(output["gate"].mean()) if output["gate"].numel() else 0.0,
                    "active_gate_fraction":float(output["gate_active_fraction"]),
                    "coordinate_finite":finite,"clipping_fraction":float(clipped.float().mean())})
            if not bounded: stable=False; reason="nonfinite_coordinate" if not finite else "coordinate_norm"; break
            x=candidate
        return x,{"stable":stable,"failure_reason":reason,"trajectory":trajectory,
            "update_scale":update_scale,"gate_override":gate_override,
            "effective_torsion_rate_scale":float(self.hparams.torsion_rate_scale if torsion_rate_scale_override is None else torsion_rate_scale_override),
            "gate_mean":sum(gate_means)/len(gate_means) if gate_means else 0.0,
            "active_gate_fraction":sum(active_fractions)/len(active_fractions) if active_fractions else 0.0,
            "torsion_rate_norm":sum(rate_norms)/len(rate_norms) if rate_norms else 0.0,
            "mean_timing":{name:sum(float(row[name]) for row in timing_rows)/len(timing_rows)
                for name in timing_rows[0]} if timing_rows else {}}

"""Explicit configuration boundary between legacy and gated motion models."""

from __future__ import annotations

from .flexbond_optimizer import FlexBondOptimizerLightningModule
from .gated_kinematic_flow import GatedKinematicFlowLightningModule


def build_motion_model(model_args: dict, **shared_args):
    arguments={**model_args,**shared_args}; motion_mode=arguments.pop("motion_mode",None)
    if motion_mode=="gated_global_torsion_kinematic":
        return GatedKinematicFlowLightningModule(motion_mode=motion_mode,**arguments)
    if motion_mode=="legacy_flexbond4d":
        arguments.setdefault("mode","flexbond4d_hybrid_optimizer")
        return FlexBondOptimizerLightningModule(**arguments)
    if motion_mode=="cartesian":
        arguments["mode"]="cartesian_optimizer"
        return FlexBondOptimizerLightningModule(**arguments)
    if motion_mode is None and "mode" in arguments:
        return FlexBondOptimizerLightningModule(**arguments)
    raise ValueError(f"Unknown motion_mode: {motion_mode!r}")

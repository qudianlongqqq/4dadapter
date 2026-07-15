"""Label-safe Stage 1 rollout and identity-bound Serial Global4D caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch_geometric.data import Dataset

from etflow.commons.global_coupled_4d_sampling import (
    checkpoint_inference_identity,
    file_sha256,
)
from etflow.data.flexbond_cache_schema import tensor_sha256, x_init_sha256
from etflow.data.flexbond_optimizer_dataset import FlexBondData
from etflow.models.flexbond_optimizer import FlexBondOptimizerLightningModule


SERIAL_CACHE_SCHEMA_VERSION = "serial-global4d-residual-v2"
LABEL_FIELDS = frozenset(
    {
        "x_ref",
        "x_ref_aligned",
        "x_ref_candidates",
        "target",
        "target_velocity",
        "u_stage2",
        "q_star",
        "q_res_star",
        "r_J_star",
        "stretch_target",
        "bending_target",
        "torsion_target",
        "selected_reference_index",
        "selected_reference_rmsd",
        "selected_ref_id",
        "rmsd_before",
        "rmsd_after",
    }
)

_CARTESIAN_INPUT_FIELDS = (
    "mol_id",
    "sample_id",
    "source_mol_id",
    "source_record_id",
    "smiles",
    "atomic_numbers",
    "node_attr",
    "edge_index",
    "bond_index",
    "edge_attr",
    "bond_type",
    "bond_is_aromatic",
    "bond_is_in_ring",
    "rotatable_bond_mask",
    "rotatable_bond_index",
    "atom_bond_influence_index",
    "num_rotatable_bonds",
    "x_init",
    "x_init_hash",
    "batch",
    "ptr",
)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _keys(record: Any) -> list[str]:
    if isinstance(record, Mapping):
        return list(record)
    keys = getattr(record, "keys", None)
    return list(keys() if callable(keys) else [])


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def label_free_cartesian_view(record: Any) -> FlexBondData:
    """Copy only the Stage 1 inference allow-list into a fresh PyG object.

    The teacher never receives the source object, so adding a label field to a
    future cache cannot silently make it visible to Cartesian rollout code.
    """

    values: dict[str, Any] = {}
    for name in _CARTESIAN_INPUT_FIELDS:
        value = _value(record, name)
        if value is not None:
            values[name] = value
    required = (
        "x_init",
        "node_attr",
        "edge_index",
        "rotatable_bond_index",
        "atom_bond_influence_index",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Cartesian inference input is missing fields: {missing}")
    leaked = sorted(LABEL_FIELDS.intersection(values))
    if leaked:
        raise AssertionError(f"Cartesian inference allow-list leaked labels: {leaked}")
    values["num_nodes"] = int(torch.as_tensor(values["x_init"]).size(0))
    view = FlexBondData(**values)
    visible_leaks = sorted(
        name for name in _keys(view) if name in LABEL_FIELDS or name.startswith("x_ref")
    )
    if visible_leaks:
        raise AssertionError(f"Cartesian inference view contains labels: {visible_leaks}")
    return view


def cartesian_sampling_identity(
    checkpoint: str | Path,
    config: str | Path,
    *,
    refinement_steps: int,
    update_scale: float,
    max_displacement: float | None,
    max_coordinate_norm: float,
    random_seed: int,
    cohort_manifest_sha256: str,
    split: str,
    code_commit: str,
    cohort_manifest_raw_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
    selection_identity: Mapping[str, Any] | None = None,
    stage2_target_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    identity = {
        "checkpoint": checkpoint_inference_identity(checkpoint),
        "config_path": str(Path(config).expanduser().resolve()),
        "config_sha256": file_sha256(config),
        "refinement_steps": int(refinement_steps),
        "update_scale": float(update_scale),
        "max_displacement": (
            None if max_displacement is None else float(max_displacement)
        ),
        "max_coordinate_norm": float(max_coordinate_norm),
        "random_seed": int(random_seed),
        "cohort_manifest_sha256": str(cohort_manifest_sha256),
        "cohort_manifest_raw_sha256": (
            None
            if cohort_manifest_raw_sha256 is None
            else str(cohort_manifest_raw_sha256)
        ),
        "split": split,
        "code_commit": str(code_commit),
        "environment": dict(environment or {}),
        "model_architecture": checkpoint_model_architecture_identity(checkpoint),
        "selection": dict(selection_identity or {}),
        "stage2_target_identity": dict(stage2_target_identity or {}),
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def checkpoint_model_architecture_identity(
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Hash model structure without binding identity to a platform path."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    hparams = payload.get("hyper_parameters")
    if not isinstance(state_dict, Mapping) or not isinstance(hparams, Mapping):
        raise ValueError("Checkpoint lacks state_dict or hyper_parameters")
    structure = {
        "model_type": (
            "cartesian_adapter"
            if hparams.get("mode") == "cartesian_optimizer"
            else hparams.get("motion_mode") or hparams.get("mode") or "unknown"
        ),
        "hyper_parameters": dict(hparams),
        "state_dict_shapes": {
            name: [str(tensor.dtype), *list(tensor.shape)]
            for name, tensor in sorted(state_dict.items())
        },
    }
    return {**structure, "architecture_sha256": _canonical_sha256(structure)}


def resolve_cartesian_teacher_selection(
    *,
    best_configs: str | Path | None = None,
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve a validation-selected teacher or an explicit Oracle candidate."""

    if checkpoint is None or config is None:
        raise ValueError("Explicit Cartesian checkpoint and config paths are required")
    if best_configs is not None:
        selection_path = Path(best_configs).expanduser().resolve()
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        configs = payload.get("configs", {})
        selected = configs.get("cartesian") or configs.get("cartesian_adapter")
        if not isinstance(selected, Mapping):
            raise ValueError("best configs has no Cartesian selection")
        if selected.get("selection_split") != "validation":
            raise ValueError("Cartesian teacher was not selected on validation")
        if bool(selected.get("test_used_for_selection")):
            raise ValueError("test split was used for Cartesian teacher selection")
        metadata = {
            "selection_mode": "frozen_validation_selection",
            "selection_path": str(selection_path),
            "selection_sha256": file_sha256(selection_path),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "validation_manifest_sha256": selected.get(
                "validation_manifest_sha256"
            ),
            "selected_checkpoint_sha256": selected.get(
                "checkpoint_file_sha256"
            ),
            "selected_config_sha256": selected.get("config_file_sha256"),
        }
    else:
        metadata = {
            "selection_mode": "explicit_validation_oracle_candidate",
            "selection_split": "validation_required",
            "test_used_for_selection": False,
        }
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    for label, path in (("checkpoint", checkpoint_path), ("config", config_path)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Cartesian {label} does not exist: {path}")
    if best_configs is not None:
        expected_checkpoint = metadata.get("selected_checkpoint_sha256")
        expected_config = metadata.get("selected_config_sha256")
        if expected_checkpoint and file_sha256(checkpoint_path) != expected_checkpoint:
            raise ValueError("Explicit checkpoint does not match validation selection SHA256")
        if expected_config and file_sha256(config_path) != expected_config:
            raise ValueError("Explicit config does not match validation selection SHA256")
    return checkpoint_path, config_path, metadata


def load_frozen_cartesian_teacher(
    checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> FlexBondOptimizerLightningModule:
    teacher = FlexBondOptimizerLightningModule.load_from_checkpoint(
        checkpoint, map_location=device
    ).to(device)
    if teacher.optimizer_mode != "cartesian_optimizer":
        raise ValueError(
            "Serial Stage 1 requires a cartesian_optimizer checkpoint; got "
            f"{teacher.optimizer_mode!r}"
        )
    teacher.eval()
    teacher.requires_grad_(False)
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise AssertionError("Cartesian teacher was not completely frozen")
    return teacher


@torch.inference_mode()
def rollout_frozen_cartesian(
    teacher: FlexBondOptimizerLightningModule,
    record: Any,
    *,
    refinement_steps: int,
    update_scale: float,
    max_displacement: float | None,
    max_coordinate_norm: float,
    device: str | torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Cartesian teacher must be eval-mode and completely frozen")
    view = label_free_cartesian_view(record).to(device)
    refined, diagnostics = teacher.refine(
        view,
        refinement_steps=int(refinement_steps),
        update_scale=float(update_scale),
        max_displacement=max_displacement,
        max_coordinate_norm=float(max_coordinate_norm),
    )
    if refined.shape != view.x_init.shape or not bool(torch.isfinite(refined).all()):
        raise FloatingPointError("Cartesian rollout returned invalid coordinates")
    return refined.detach().cpu(), dict(diagnostics)


def _graph_payload(record: Any) -> dict[str, Any]:
    names = (
        "mol_id",
        "sample_id",
        "source_mol_id",
        "source_record_id",
        "smiles",
        "atomic_numbers",
        "node_attr",
        "edge_index",
        "edge_attr",
        "bond_type",
        "bond_is_aromatic",
        "bond_is_in_ring",
        "rotatable_bond_mask",
        "rotatable_bond_index",
        "atom_bond_influence_index",
        "num_rotatable_bonds",
    )
    return {name: _value(record, name) for name in names if _value(record, name) is not None}


def build_stage2_training_record(
    source: Any,
    x_cart: Tensor,
    *,
    teacher_sampling_identity: Mapping[str, Any],
    original_manifest_identity: str,
    split: str,
    pilot_manifest_identity: str | None = None,
    stage2_targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("Stage 2 training caches may only be built from train or val")
    x_init = torch.as_tensor(_value(source, "x_init"), dtype=torch.float32).cpu()
    x_ref = _value(source, "x_ref_aligned")
    if x_ref is None:
        raise ValueError("Stage 2 training cache requires x_ref_aligned")
    x_ref = torch.as_tensor(x_ref, dtype=torch.float32).cpu()
    x_cart = torch.as_tensor(x_cart, dtype=torch.float32).detach().cpu()
    if x_cart.shape != x_init.shape or x_ref.shape != x_init.shape:
        raise ValueError("x_init, x_cart, and x_ref_aligned shapes must match")
    if not bool(torch.isfinite(torch.stack((x_init, x_cart, x_ref))).all()):
        raise ValueError("Stage 2 coordinates must be finite")
    if torch.equal(x_cart, x_ref):
        raise ValueError("x_cart is an exact copy of x_ref_aligned")
    atomic_numbers = torch.as_tensor(_value(source, "atomic_numbers"), dtype=torch.long)
    graph = _graph_payload(source)
    source_mol_id = str(
        _value(source, "source_mol_id", _value(source, "mol_id", ""))
    )
    graph["cache_mol_id"] = str(_value(source, "mol_id", source_mol_id))
    graph["mol_id"] = source_mol_id
    payload = {
        **graph,
        "source_split": split,
        "source_manifest_sha": str(original_manifest_identity),
        "pilot_manifest_sha": str(
            pilot_manifest_identity or original_manifest_identity
        ),
        "x_init": x_init,
        "x_init_hash": str(
            _value(source, "x_init_hash", x_init_sha256(x_init, atomic_numbers))
        ),
        "x_cart": x_cart,
        "x_cart_sha256": tensor_sha256(x_cart),
        "x_ref_aligned": x_ref,
        "u_stage2": x_ref - x_cart,
        "num_atoms": int(x_init.size(0)),
        "num_edges": int(torch.as_tensor(graph["edge_index"]).size(1)),
        "num_joints": int(
            torch.as_tensor(graph["rotatable_bond_index"]).size(1)
        ),
        "flexibility_cohort": (
            "low"
            if int(torch.as_tensor(graph["rotatable_bond_index"]).size(1)) <= 2
            else (
                "medium"
                if int(torch.as_tensor(graph["rotatable_bond_index"]).size(1)) <= 5
                else "high"
            )
        ),
        "teacher_sampling_identity": dict(teacher_sampling_identity),
        "teacher_sampling_identity_sha256": str(
            teacher_sampling_identity.get("identity_sha256")
            or _canonical_sha256(teacher_sampling_identity)
        ),
        "original_manifest_identity": str(original_manifest_identity),
        "split": split,
        "stage2_cache_schema_version": SERIAL_CACHE_SCHEMA_VERSION,
        **dict(stage2_targets or {}),
    }
    validate_stage2_training_record(payload)
    return payload


def _validate_common(record: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    if record.get("stage2_cache_schema_version") != SERIAL_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported Serial Global4D cache schema")
    x_cart = torch.as_tensor(record.get("x_cart"), dtype=torch.float32)
    atomic_numbers = torch.as_tensor(record.get("atomic_numbers"), dtype=torch.long).view(-1)
    if tuple(x_cart.shape) != (atomic_numbers.numel(), 3):
        raise ValueError("x_cart shape does not match ordered atoms")
    if not bool(torch.isfinite(x_cart).all()):
        raise ValueError("x_cart contains NaN or Inf")
    if str(record.get("x_cart_sha256")) != tensor_sha256(x_cart):
        raise ValueError("x_cart_sha256 mismatch")
    identity = record.get("teacher_sampling_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("teacher_sampling_identity is required")
    expected = identity.get("identity_sha256") or _canonical_sha256(identity)
    if str(record.get("teacher_sampling_identity_sha256")) != str(expected):
        raise ValueError("teacher sampling identity mismatch")
    if not str(record.get("original_manifest_identity", "")).strip():
        raise ValueError("original_manifest_identity is required")
    return x_cart, atomic_numbers


def validate_stage2_training_record(
    record: Mapping[str, Any], *, require_targets: bool = False
) -> dict[str, Tensor]:
    x_cart, atomic_numbers = _validate_common(record)
    if record.get("split") not in {"train", "val"}:
        raise ValueError("Stage 2 training record split must be train or val")
    x_ref = torch.as_tensor(record.get("x_ref_aligned"), dtype=torch.float32)
    if x_ref.shape != x_cart.shape or not bool(torch.isfinite(x_ref).all()):
        raise ValueError("x_ref_aligned is invalid")
    if torch.equal(x_cart, x_ref):
        raise ValueError("x_cart is an exact copy of x_ref_aligned")
    u_stage2 = x_ref - x_cart
    persisted_u = record.get("u_stage2")
    if persisted_u is not None and not torch.allclose(
        torch.as_tensor(persisted_u), u_stage2, atol=1.0e-7, rtol=1.0e-7
    ):
        raise ValueError("u_stage2 does not equal x_ref_aligned - x_cart")
    checked = {
        "x_cart": x_cart,
        "x_ref_aligned": x_ref,
        "u_stage2": u_stage2,
        "atomic_numbers": atomic_numbers,
    }
    target_fields = (
        "target_time",
        "q_res_star",
        "r_J_star",
        "residual_norm",
        "projected_residual_norm",
        "projection_energy_ratio",
        "jacobian_rank",
        "solver_mode",
        "solver_fallback",
    )
    missing = [name for name in target_fields if name not in record]
    if require_targets and missing:
        raise ValueError(f"Stage 2 target record is missing fields: {missing}")
    if not missing:
        q_star = torch.as_tensor(record["q_res_star"], dtype=torch.float32)
        r_star = torch.as_tensor(record["r_J_star"], dtype=torch.float32)
        if q_star.ndim != 2 or q_star.size(1) != 4:
            raise ValueError("q_res_star must have shape [J, 4]")
        if r_star.shape != x_cart.shape:
            raise ValueError("r_J_star shape must match x_cart")
        if not bool(torch.isfinite(q_star).all() and torch.isfinite(r_star).all()):
            raise ValueError("Stage 2 Oracle targets contain NaN or Inf")
        checked.update({"q_res_star": q_star, "r_J_star": r_star})
    return checked


def validate_stage2_inference_record(record: Mapping[str, Any]) -> dict[str, Tensor]:
    leaked = sorted(
        name
        for name in record
        if name in LABEL_FIELDS or name.startswith("x_ref") or name.startswith("target")
    )
    if leaked:
        raise ValueError(f"Stage 2 inference record contains labels: {leaked}")
    x_cart, atomic_numbers = _validate_common(record)
    return {"x_cart": x_cart, "atomic_numbers": atomic_numbers}


def assert_teacher_identity(
    record: Mapping[str, Any], expected_identity: Mapping[str, Any]
) -> None:
    expected = str(
        expected_identity.get("identity_sha256") or _canonical_sha256(expected_identity)
    )
    actual = str(record.get("teacher_sampling_identity_sha256", ""))
    if actual != expected:
        raise ValueError(
            "Stage 2 cache belongs to a different Cartesian teacher or sampling command"
        )


class SerialGlobal4DResidualDataset(Dataset):
    """Read identity-bound Stage 2 records without touching the source cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        expected_teacher_identity: Mapping[str, Any] | None = None,
        inference: bool = False,
        require_targets: bool = True,
    ) -> None:
        super().__init__()
        root = Path(cache_dir).expanduser()
        if (root / split).is_dir():
            root = root / split
        self.files = sorted(root.glob("*.pt"))
        if not self.files:
            raise ValueError(f"No Serial Global4D cache files found in {root}")
        self.expected_teacher_identity = expected_teacher_identity
        self.inference = bool(inference)
        self.require_targets = bool(require_targets)

    def len(self) -> int:
        return len(self.files)

    def get(self, index: int) -> FlexBondData:
        record = torch.load(self.files[index], map_location="cpu", weights_only=False)
        if not isinstance(record, Mapping):
            raise TypeError("Serial cache payload must be a mapping")
        values = (
            validate_stage2_inference_record(record)
            if self.inference
            else validate_stage2_training_record(
                record, require_targets=self.require_targets
            )
        )
        if self.expected_teacher_identity is not None:
            assert_teacher_identity(record, self.expected_teacher_identity)
        fields = _graph_payload(record)
        fields["x_cart"] = values["x_cart"]
        fields["x_init"] = values["x_cart"]
        if not self.inference:
            fields["x_ref_aligned"] = values["x_ref_aligned"]
            fields["u_stage2"] = values["u_stage2"]
            if "q_res_star" in values:
                fields["q_res_star"] = values["q_res_star"]
                fields["r_J_star"] = values["r_J_star"]
            for name in (
                "target_time",
                "residual_norm",
                "projected_residual_norm",
                "projection_energy_ratio",
                "stretch_target",
                "bending_target",
                "torsion_target",
                "jacobian_rank",
                "condition_number",
                "solver_mode",
                "solver_fallback",
                "target_reconstruction_error",
                "flexibility_cohort",
            ):
                if name in record:
                    fields[name] = record[name]
        fields["num_nodes"] = int(values["x_cart"].size(0))
        fields["teacher_sampling_identity_sha256"] = record[
            "teacher_sampling_identity_sha256"
        ]
        fields["original_manifest_identity"] = record["original_manifest_identity"]
        return FlexBondData(**fields)

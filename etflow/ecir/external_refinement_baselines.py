"""Identity-bound external conformer refinement baselines.

The optimizers in this module are evaluation-only.  They consume the frozen
ET-Flow source coordinates and molecular graph, never reference coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "frozen_holdout_records_read": 0,
    "minimal_validity_target_test_used": False,
    "parameter_selection_from_formal_test": False,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def coordinate_sha256(value: Any) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    descriptor = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return hashlib.sha256(descriptor + tensor.numpy().tobytes()).hexdigest()


def _mapping(record: Mapping[str, Any]) -> tuple[int, ...]:
    mapping = tuple(int(value) for value in record["_formal_cache_to_rdkit"])
    if len(mapping) != int(record["num_atoms"]):
        raise ValueError("cache/RDKit atom mapping length changed")
    if len(set(mapping)) != len(mapping):
        raise ValueError("cache/RDKit atom mapping is not one-to-one")
    return mapping


def mol_from_frozen_record(record: Mapping[str, Any], coordinates: Any) -> Chem.Mol:
    """Clone the frozen RDKit graph and install coordinates without embedding."""

    frozen = record.get("_formal_rdkit_mol")
    if frozen is None:
        raise ValueError("frozen RDKit molecule is unavailable")
    mol = Chem.Mol(frozen)
    mapping = _mapping(record)
    xyz = np.asarray(torch.as_tensor(coordinates).detach().cpu(), dtype=np.float64)
    if xyz.shape != (len(mapping), 3) or not np.isfinite(xyz).all():
        raise ValueError("source coordinates are invalid")
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for cache_index, rdkit_index in enumerate(mapping):
        conformer.SetAtomPosition(rdkit_index, xyz[cache_index].tolist())
    mol.AddConformer(conformer, assignId=True)
    return mol


def coordinates_in_cache_order(mol: Chem.Mol, record: Mapping[str, Any]) -> torch.Tensor:
    conformer = mol.GetConformer()
    values = [list(conformer.GetAtomPosition(rdkit_index)) for rdkit_index in _mapping(record)]
    return torch.tensor(values, dtype=torch.float32)


def ordered_atom_identity(mol: Chem.Mol, record: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    return [
        (
            int(mol.GetAtomWithIdx(rdkit_index).GetAtomicNum()),
            int(mol.GetAtomWithIdx(rdkit_index).GetIsotope()),
            int(mol.GetAtomWithIdx(rdkit_index).GetFormalCharge()),
            int(mol.GetAtomWithIdx(rdkit_index).GetNumRadicalElectrons()),
        )
        for rdkit_index in _mapping(record)
    ]


def ordered_atom_identity_sha256(mol: Chem.Mol, record: Mapping[str, Any]) -> str:
    return canonical_sha256(ordered_atom_identity(mol, record))


def topology_identity_sha256(mol: Chem.Mol, record: Mapping[str, Any]) -> str:
    mapping = _mapping(record)
    inverse = {rdkit_index: cache_index for cache_index, rdkit_index in enumerate(mapping)}
    bonds = []
    for bond in mol.GetBonds():
        begin, end = inverse[bond.GetBeginAtomIdx()], inverse[bond.GetEndAtomIdx()]
        bonds.append(
            (
                min(begin, end),
                max(begin, end),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic()),
            )
        )
    return canonical_sha256({"atoms": ordered_atom_identity(mol, record), "bonds": sorted(bonds)})


def validate_atom_identity(mol: Chem.Mol, record: Mapping[str, Any]) -> bool:
    expected = [int(value) for value in torch.as_tensor(record["atomic_numbers"]).tolist()]
    observed = [value[0] for value in ordered_atom_identity(mol, record)]
    return observed == expected and mol.GetNumAtoms() == len(expected)


def validate_topology_identity(before: Chem.Mol, after: Chem.Mol, record: Mapping[str, Any]) -> bool:
    return topology_identity_sha256(before, record) == topology_identity_sha256(after, record)


def validate_coordinates(coordinates: Any, atom_count: int) -> bool:
    array = np.asarray(torch.as_tensor(coordinates).detach().cpu(), dtype=np.float64)
    return array.shape == (int(atom_count), 3) and bool(np.isfinite(array).all())


def derive_total_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def derive_unpaired_electrons(mol: Chem.Mol) -> int:
    return int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))


def _chirality_signatures(coordinates: torch.Tensor, record: Mapping[str, Any]) -> tuple[int, ...]:
    xyz = torch.as_tensor(coordinates, dtype=torch.float64)
    signatures = []
    for center, first, second, third in record.get("_formal_chiral_center_quads", ()):
        vectors = xyz[[first, second, third]] - xyz[int(center)]
        volume = float(torch.linalg.det(vectors))
        signatures.append(0 if abs(volume) < 1.0e-8 else (1 if volume > 0 else -1))
    return tuple(signatures)


def validate_chirality(source: Any, refined: Any, record: Mapping[str, Any]) -> bool:
    before = _chirality_signatures(torch.as_tensor(source), record)
    after = _chirality_signatures(torch.as_tensor(refined), record)
    return len(before) == len(after) and all(left != 0 and left == right for left, right in zip(before, after))


@dataclass
class ExternalRefinementResult:
    method: str
    method_version: str
    source_coordinates: torch.Tensor
    refined_coordinates: torch.Tensor
    success: bool
    converged: bool
    fallback_to_source: bool
    failure_reason: str | None
    runtime_seconds: float
    iteration_count: int | None = None
    cycle_count: int | None = None
    initial_native_energy: float | None = None
    final_native_energy: float | None = None
    native_energy_delta: float | None = None
    atom_count_before: int = 0
    atom_count_after: int = 0
    atom_order_verified: bool = False
    topology_verified: bool = False
    chirality_verified: bool = False
    total_charge: int = 0
    unpaired_electrons: int = 0
    timeout: bool = False
    unsupported: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        payload = asdict(self)
        source = self.source_coordinates.detach().cpu().to(torch.float32)
        refined = self.refined_coordinates.detach().cpu().to(torch.float32)
        displacement = torch.linalg.vector_norm(refined - source, dim=-1)
        payload.update(
            {
                "source_coordinates": source,
                "refined_coordinates": refined,
                "safe_coordinates": refined,
                "mean_atom_displacement": float(displacement.mean()),
                "max_atom_displacement": float(displacement.max()),
                "accepted": bool(self.success and not self.fallback_to_source),
                "rollback": bool(self.fallback_to_source),
                "backtracking_decision": {
                    "accepted": bool(self.success and not self.fallback_to_source),
                    "rolled_back": bool(self.fallback_to_source),
                    "reasons": [] if not self.failure_reason else [self.failure_reason],
                },
                "solver_diagnostics": {
                    "failure_count": int(not self.success),
                    "bond_contribution": 0.0,
                    "angle_contribution": 0.0,
                },
            }
        )
        return payload


def fallback_to_source(
    method: str,
    version: str,
    source: Any,
    reason: str,
    *,
    started: float,
    mol: Chem.Mol | None = None,
    timeout: bool = False,
    unsupported: bool = False,
    diagnostics: Mapping[str, Any] | None = None,
) -> ExternalRefinementResult:
    coordinates = torch.as_tensor(source).detach().cpu().to(torch.float32).clone()
    atom_count = int(coordinates.shape[0])
    return ExternalRefinementResult(
        method=method,
        method_version=version,
        source_coordinates=coordinates,
        refined_coordinates=coordinates.clone(),
        success=False,
        converged=False,
        fallback_to_source=True,
        failure_reason=str(reason),
        runtime_seconds=time.perf_counter() - started,
        atom_count_before=atom_count,
        atom_count_after=atom_count,
        atom_order_verified=mol is not None,
        topology_verified=mol is not None,
        chirality_verified=True,
        total_charge=derive_total_charge(mol) if mol is not None else 0,
        unpaired_electrons=derive_unpaired_electrons(mol) if mol is not None else 0,
        timeout=timeout,
        unsupported=unsupported,
        diagnostics=dict(diagnostics or {}),
    )


def _validated_success(
    *,
    method: str,
    version: str,
    source: torch.Tensor,
    refined: torch.Tensor,
    before_mol: Chem.Mol,
    after_mol: Chem.Mol,
    record: Mapping[str, Any],
    started: float,
    initial_energy: float | None,
    final_energy: float | None,
    iteration_count: int | None = None,
    cycle_count: int | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> ExternalRefinementResult:
    atom_ok = validate_atom_identity(after_mol, record)
    topology_ok = validate_topology_identity(before_mol, after_mol, record)
    coordinate_ok = validate_coordinates(refined, source.shape[0])
    chirality_ok = validate_chirality(source, refined, record)
    failures = []
    if not atom_ok:
        failures.append("atom_identity_changed")
    if not topology_ok:
        failures.append("topology_identity_changed")
    if not coordinate_ok:
        failures.append("nonfinite_or_shape_changed")
    if not chirality_ok:
        failures.append("chirality_changed_or_degenerate")
    if failures:
        return fallback_to_source(
            method, version, source, "+".join(failures), started=started, mol=before_mol,
            diagnostics=diagnostics,
        )
    delta = None
    if initial_energy is not None and final_energy is not None:
        delta = float(final_energy - initial_energy)
    return ExternalRefinementResult(
        method=method,
        method_version=version,
        source_coordinates=source.clone(),
        refined_coordinates=refined.clone(),
        success=True,
        converged=True,
        fallback_to_source=False,
        failure_reason=None,
        runtime_seconds=time.perf_counter() - started,
        iteration_count=iteration_count,
        cycle_count=cycle_count,
        initial_native_energy=initial_energy,
        final_native_energy=final_energy,
        native_energy_delta=delta,
        atom_count_before=int(source.shape[0]),
        atom_count_after=int(refined.shape[0]),
        atom_order_verified=atom_ok,
        topology_verified=topology_ok,
        chirality_verified=chirality_ok,
        total_charge=derive_total_charge(before_mol),
        unpaired_electrons=derive_unpaired_electrons(before_mol),
        diagnostics=dict(diagnostics or {}),
    )


def refine_with_mmff94s(
    record: Mapping[str, Any], source_coordinates: Any, config: Mapping[str, Any]
) -> ExternalRefinementResult:
    started = time.perf_counter()
    source = torch.as_tensor(source_coordinates).detach().cpu().to(torch.float32)
    version = f"RDKit-{rdBase.rdkitVersion}/MMFF94s"
    try:
        before = mol_from_frozen_record(record, source)
        if not validate_atom_identity(before, record):
            return fallback_to_source("MMFF94S", version, source, "atom_identity_changed", started=started)
        if not AllChem.MMFFHasAllMoleculeParams(before):
            return fallback_to_source(
                "MMFF94S", version, source, "mmff_parameters_unavailable", started=started,
                mol=before, unsupported=True,
            )
        props = AllChem.MMFFGetMoleculeProperties(before, mmffVariant="MMFF94s")
        if props is None:
            return fallback_to_source(
                "MMFF94S", version, source, "mmff94s_properties_unavailable", started=started,
                mol=before, unsupported=True,
            )
        initial_ff = AllChem.MMFFGetMoleculeForceField(
            before, props, nonBondedThresh=float(config["nonbonded_threshold"]), confId=0,
            ignoreInterfragInteractions=False,
        )
        if initial_ff is None:
            return fallback_to_source("MMFF94S", version, source, "mmff_forcefield_unavailable", started=started, mol=before, unsupported=True)
        initial_energy = float(initial_ff.CalcEnergy())
        after = Chem.Mol(before)
        return_code = int(
            AllChem.MMFFOptimizeMolecule(
                after,
                mmffVariant="MMFF94s",
                maxIters=int(config["max_iterations"]),
                nonBondedThresh=float(config["nonbonded_threshold"]),
                ignoreInterfragInteractions=False,
            )
        )
        refined = coordinates_in_cache_order(after, record)
        final_props = AllChem.MMFFGetMoleculeProperties(after, mmffVariant="MMFF94s")
        final_ff = AllChem.MMFFGetMoleculeForceField(
            after, final_props, nonBondedThresh=float(config["nonbonded_threshold"]), confId=0,
            ignoreInterfragInteractions=False,
        )
        final_energy = float(final_ff.CalcEnergy()) if final_ff is not None else math.nan
        if return_code != 0:
            reason = "mmff_more_iterations_required" if return_code == 1 else "mmff_setup_failure"
            return fallback_to_source(
                "MMFF94S", version, source, reason, started=started, mol=before,
                diagnostics={"return_code": return_code, "last_energy": final_energy},
            )
        if not math.isfinite(initial_energy) or not math.isfinite(final_energy):
            return fallback_to_source("MMFF94S", version, source, "nonfinite_energy", started=started, mol=before)
        return _validated_success(
            method="MMFF94S", version=version, source=source, refined=refined,
            before_mol=before, after_mol=after, record=record, started=started,
            initial_energy=initial_energy, final_energy=final_energy,
            iteration_count=None, diagnostics={"return_code": return_code, "mmff_variant": "MMFF94s"},
        )
    except BaseException as error:
        return fallback_to_source("MMFF94S", version, source, f"mmff_exception:{type(error).__name__}:{error}", started=started)


def _write_xyz(path: Path, elements: Sequence[str], coordinates: torch.Tensor) -> None:
    lines = [str(len(elements)), "frozen ET-Flow source; no reference geometry"]
    for element, coordinate in zip(elements, coordinates.tolist(), strict=True):
        lines.append(f"{element} {coordinate[0]:.12f} {coordinate[1]:.12f} {coordinate[2]:.12f}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(path)


def _read_xyz(path: Path) -> tuple[list[str], torch.Tensor]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    count = int(lines[0].strip())
    rows = [line.split() for line in lines[2 : 2 + count]]
    if len(rows) != count or any(len(row) < 4 for row in rows):
        raise ValueError("xTB XYZ output is incomplete")
    elements = [row[0] for row in rows]
    coordinates = torch.tensor([[float(v) for v in row[1:4]] for row in rows], dtype=torch.float32)
    return elements, coordinates


def windows_path_to_wsl(path: str | Path) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def build_xtb_command(
    *, input_name: str, workdir: str | Path, total_charge: int,
    unpaired_electrons: int, config: Mapping[str, Any]
) -> list[str]:
    executable = windows_path_to_wsl(config["xtb_executable"])
    return [
        str(config.get("wsl_executable", "wsl.exe")), "-d", str(config["wsl_distribution"]),
        "--cd", windows_path_to_wsl(workdir), "--exec", "/usr/bin/env",
        f"OMP_NUM_THREADS={int(config['omp_threads_per_process'])}",
        f"MKL_NUM_THREADS={int(config['omp_threads_per_process'])}",
        f"OPENBLAS_NUM_THREADS={int(config['omp_threads_per_process'])}",
        executable, input_name, "--gfn", "2", "--opt", str(config["optimization_level"]),
        "--cycles", str(int(config["maximum_cycles"])), "--chrg", str(int(total_charge)),
        "--uhf", str(int(unpaired_electrons)),
    ]


def _xtb_energies(output: str) -> tuple[float | None, float | None]:
    # 6.7.1 prints a detailed ``:: total energy`` block before optimization
    # and again for the final structure.  Prefer those values; the boxed
    # ``TOTAL ENERGY`` footer contains only the final energy.
    values = [
        float(value)
        for value in re.findall(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh", output, re.I)
    ]
    if len(values) < 2:
        values = [
            float(value)
            for value in re.findall(r"^\s*\*\s*total energy\s*:\s*(-?\d+\.\d+)", output, re.I | re.M)
        ]
    return (values[0], values[-1]) if values else (None, None)


def refine_with_gfn2_xtb(
    record: Mapping[str, Any], source_coordinates: Any, config: Mapping[str, Any],
    workdir: str | Path,
) -> ExternalRefinementResult:
    started = time.perf_counter()
    source = torch.as_tensor(source_coordinates).detach().cpu().to(torch.float32)
    version = str(config["xtb_version"])
    work = Path(workdir).resolve()
    try:
        before = mol_from_frozen_record(record, source)
        if not validate_atom_identity(before, record):
            return fallback_to_source("GFN2_XTB", version, source, "atom_identity_changed", started=started)
        work.mkdir(parents=True, exist_ok=True)
        elements = [Chem.GetPeriodicTable().GetElementSymbol(value[0]) for value in ordered_atom_identity(before, record)]
        _write_xyz(work / "input.xyz", elements, source)
        charge = derive_total_charge(before)
        unpaired = derive_unpaired_electrons(before)
        command = build_xtb_command(
            input_name="input.xyz", workdir=work, total_charge=charge,
            unpaired_electrons=unpaired, config=config,
        )
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=float(config["timeout_seconds"]),
                check=False, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as error:
            return fallback_to_source(
                "GFN2_XTB", version, source, "timeout", started=started, mol=before, timeout=True,
                diagnostics={"command": command, "stdout_tail": str(error.stdout or "")[-4000:], "stderr_tail": str(error.stderr or "")[-4000:]},
            )
        output = (process.stdout or "") + "\n" + (process.stderr or "")
        (work / "xtb.stdout.txt").write_text(output, encoding="utf-8")
        not_converged = (work / "NOT_CONVERGED").exists() or "NOT CONVERGED" in output.upper()
        converged_text = "GEOMETRY OPTIMIZATION CONVERGED" in output.upper()
        if process.returncode != 0:
            return fallback_to_source("GFN2_XTB", version, source, f"process_return_code:{process.returncode}", started=started, mol=before, diagnostics={"command": command, "output_tail": output[-4000:]})
        if not_converged or not converged_text:
            return fallback_to_source("GFN2_XTB", version, source, "xtb_not_converged", started=started, mol=before, diagnostics={"command": command, "output_tail": output[-4000:]})
        optimized = work / "xtbopt.xyz"
        if not optimized.exists():
            return fallback_to_source("GFN2_XTB", version, source, "missing_xtbopt_xyz", started=started, mol=before)
        observed_elements, refined = _read_xyz(optimized)
        if observed_elements != elements:
            return fallback_to_source("GFN2_XTB", version, source, "element_order_changed", started=started, mol=before)
        after = mol_from_frozen_record(record, refined)
        initial_energy, final_energy = _xtb_energies(output)
        if initial_energy is None or final_energy is None or not all(math.isfinite(v) for v in (initial_energy, final_energy)):
            return fallback_to_source("GFN2_XTB", version, source, "nonfinite_or_missing_energy", started=started, mol=before)
        cycle_values = [int(value) for value in re.findall(r"CYCLE\s+(\d+)", output, re.I)]
        cycles = max(cycle_values) if cycle_values else None
        result = _validated_success(
            method="GFN2_XTB", version=version, source=source, refined=refined,
            before_mol=before, after_mol=after, record=record, started=started,
            initial_energy=initial_energy, final_energy=final_energy, cycle_count=cycles,
            diagnostics={"command": command, "return_code": process.returncode},
        )
        if result.success and bool(config.get("cleanup_successful_workdirs", True)):
            shutil.rmtree(work, ignore_errors=True)
        return result
    except BaseException as error:
        return fallback_to_source(
            "GFN2_XTB", version, source, f"xtb_exception:{type(error).__name__}:{error}",
            started=started,
        )


def serialize_refinement_record(
    result: ExternalRefinementResult, *, record_index: int, sample_id: str,
    molecule_id: str, source_record_sha256: str, ordered_atom_identity_sha256: str,
    topology_identity_sha256: str, method_config_sha256: str,
) -> dict[str, Any]:
    return {
        "record_index": int(record_index),
        "sample_id": str(sample_id),
        "molecule_id": str(molecule_id),
        "source_record_sha256": str(source_record_sha256),
        "ordered_atom_identity_sha256": str(ordered_atom_identity_sha256),
        "topology_identity_sha256": str(topology_identity_sha256),
        "source_coordinate_sha256": coordinate_sha256(result.source_coordinates),
        "method_config_sha256": str(method_config_sha256),
        **result.as_record(),
        **ISOLATION,
    }

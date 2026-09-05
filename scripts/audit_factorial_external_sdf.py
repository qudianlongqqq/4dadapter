"""Read-only N-record PoseBusters/Validity3D probe using frozen evaluator definitions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
import psutil
import yaml
from rdkit import Chem

ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
WORKER = ROOT / "scripts/evaluate_sixs_musigma_external.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state() -> dict[str, float | int]:
    proc = psutil.Process()
    cpu = proc.cpu_times()
    io = proc.io_counters()
    return {
        "wall": time.perf_counter(),
        "cpu_seconds": float(cpu.user + cpu.system),
        "rss_bytes": int(proc.memory_info().rss),
        "read_bytes": int(io.read_bytes),
        "write_bytes": int(io.write_bytes),
        "threads": int(proc.num_threads()),
    }


def measured(fn):
    before = state()
    value = fn()
    after = state()
    return value, {f"delta_{key}": after[key] - before[key] for key in before} | {"before": before, "after": after}


def load_worker():
    spec = importlib.util.spec_from_file_location("factorial_external_worker_audit", WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = load_worker()
    supplier = Chem.SDMolSupplier(str(args.sdf), removeHs=False, sanitize=True)
    molecules = [supplier[index] for index in range(args.n)]
    if any(molecule is None for molecule in molecules):
        raise RuntimeError("subset contains an unreadable molecule")
    ids = [molecule.GetProp("_Name") for molecule in molecules]
    subset = args.output_dir / "subset.sdf"
    writer = Chem.SDWriter(str(subset))
    for molecule in molecules:
        writer.write(molecule)
    writer.close()

    import posebusters
    pb_root = Path(posebusters.__file__).parent
    installed = yaml.safe_load((pb_root / "config/mol_fast.yml").read_text(encoding="utf-8"))
    runtime = copy.deepcopy(installed)
    for module in runtime.get("modules", []):
        if module.get("function") == "energy_ratio":
            module.setdefault("parameters", {})["num_threads"] = 1
    from posebusters import PoseBusters
    chosen = worker.selected_columns(installed)

    def run_pb():
        frame = PoseBusters(config=runtime, max_workers=1, chunk_size=50).bust(str(subset), None, None, full_report=True).reset_index()
        order = {sample: index for index, sample in enumerate(ids)}
        frame["_order"] = frame.molecule.astype(str).map(order)
        frame = frame.sort_values("_order", kind="stable").reset_index(drop=True)
        if len(frame) != args.n or frame.molecule.astype(str).tolist() != ids or not set(chosen).issubset(frame.columns):
            raise RuntimeError("PoseBusters subset identity/schema failure")
        frame["record_id"] = ids
        frame["PB"] = frame[chosen].fillna(False).astype(bool).all(axis=1)
        return frame

    pb, pb_resource = measured(run_pb)
    pb_path = args.output_dir / "posebusters.parquet"
    pb.to_parquet(pb_path, index=False)

    if sha256(worker.OFFICIAL_ADAPTER) != worker.GENBENCH_ADAPTER_SHA:
        raise RuntimeError("GenBench3D adapter hash changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worker.GENBENCH_REPO, text=True).strip() != worker.GENBENCH_COMMIT:
        raise RuntimeError("GenBench3D commit changed")
    for path, expected in (
        (worker.LIGBOUNDCONF, worker.REFERENCE_SDF_SHA),
        (worker.REFERENCE_ROOT / "LigBoundConf_geometry_values.p", worker.REFERENCE_VALUE_SHA),
        (worker.REFERENCE_ROOT / "LigBoundConf_geometry_kernel_densities.p", worker.REFERENCE_KERNEL_SHA),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"GenBench3D reference identity changed: {path}")
    sys.modules.setdefault("gemmi", types.ModuleType("gemmi"))
    spec = importlib.util.spec_from_file_location("factorial_genbench_external_audit", worker.OFFICIAL_ADAPTER)
    official = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(official)
    official.METHODS = ("J0-R1-AUDIT",)
    SDFSource, ReferenceGeometry, Validity3D = official.import_official_evaluator()

    def run_v3d():
        reference = ReferenceGeometry(
            source=SDFSource(str(worker.LIGBOUNDCONF), name="LigBoundConf", removeHs=False),
            root=str(worker.REFERENCE_ROOT), minimum_pattern_values=50,
            use_generalized_patterns=False,
        )
        validity = Validity3D(
            reference_geometry=reference, q_value_threshold=.001,
            steric_clash_safety_ratio=.75, maximum_ring_plane_distance=.1,
            include_torsions=False, consider_hydrogens=False,
        )
        subset_supplier = Chem.SDMolSupplier(str(subset), removeHs=False, sanitize=True)
        rows = []
        for index, sample_id in enumerate(ids):
            molecule = subset_supplier[index]
            output = official.evaluate_record_group(validity, [molecule], index)
            if len(output) != 1:
                raise RuntimeError("Validity3D group denominator changed")
            output[0]["record_id"] = sample_id
            rows.append(output[0])
        frame = pd.DataFrame(rows)
        if len(frame) != args.n or frame.record_id.astype(str).tolist() != ids:
            raise RuntimeError("Validity3D subset identity failure")
        return frame

    v3d, v3d_resource = measured(run_v3d)
    v3d_path = args.output_dir / "validity3d.parquet"
    v3d.to_parquet(v3d_path, index=False)
    summary = {
        "schema_version": "sixs-factorial-external-sdf-probe-v1",
        "python": sys.version,
        "executable": sys.executable,
        "n": args.n,
        "source_sdf_sha256": sha256(args.sdf),
        "subset_sdf_sha256": sha256(subset),
        "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "posebusters": {"resource": pb_resource, "aggregate": float(pb.PB.mean()), "sha256": sha256(pb_path), "columns": list(pb.columns)},
        "validity3d": {"resource": v3d_resource, "aggregate": float(v3d.validity3d.mean()), "sha256": sha256(v3d_path), "columns": list(v3d.columns)},
        "torch_imported": "torch" in sys.modules,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

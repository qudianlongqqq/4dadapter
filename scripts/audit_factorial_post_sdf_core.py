"""Read-only resource probe for frozen factorial SDF post-processing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def gpu_snapshot() -> dict[str, object]:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split(",")
        return {"utilization_percent": float(raw[0]), "memory_used_mib": float(raw[1]), "memory_total_mib": float(raw[2])}
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def state() -> dict[str, object]:
    proc = psutil.Process()
    cpu = proc.cpu_times()
    io = proc.io_counters()
    torch_cuda_allocated = None
    torch_cuda_reserved = None
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch_cuda_allocated = int(torch.cuda.memory_allocated())
            torch_cuda_reserved = int(torch.cuda.memory_reserved())
    return {
        "wall": time.perf_counter(),
        "cpu_seconds": float(cpu.user + cpu.system),
        "rss_bytes": int(proc.memory_info().rss),
        "read_bytes": int(io.read_bytes),
        "write_bytes": int(io.write_bytes),
        "threads": int(proc.num_threads()),
        "torch_cuda_allocated_bytes": torch_cuda_allocated,
        "torch_cuda_reserved_bytes": torch_cuda_reserved,
        "gpu": gpu_snapshot(),
    }


def delta(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    numeric = ("wall", "cpu_seconds", "rss_bytes", "read_bytes", "write_bytes", "threads")
    result = {f"delta_{key}": after[key] - before[key] for key in numeric}
    result["before"] = before
    result["after"] = after
    return result


def run_stage(name: str, fn, stages: dict[str, object]):
    before = state()
    value = fn()
    after = state()
    stages[name] = delta(before, after)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / f"scratch_{args.output.stem}"
    scratch.mkdir(parents=True, exist_ok=True)
    stages: dict[str, object] = {}

    def imports():
        loaded = {}
        for name in ("torch", "rdkit", "numpy", "pandas", "scipy", "pyarrow"):
            module = importlib.import_module(name)
            loaded[name] = getattr(module, "__version__", "UNKNOWN")
        return loaded

    versions = run_stage("01_import_libraries", imports, stages)
    torch = sys.modules["torch"]
    np = sys.modules["numpy"]
    pd = sys.modules["pandas"]
    Chem = importlib.import_module("rdkit.Chem")

    supplier = run_stage(
        "02_open_sdf",
        lambda: Chem.SDMolSupplier(str(args.sdf), removeHs=False, sanitize=True),
        stages,
    )
    first = run_stage("03_parse_first_molecule", lambda: supplier[0], stages)
    if first is None:
        raise RuntimeError("first frozen SDF molecule failed parsing")
    molecules = run_stage("04_parse_requested_molecules", lambda: [supplier[i] for i in range(args.n)], stages)
    if any(molecule is None for molecule in molecules):
        raise RuntimeError("frozen SDF parsing returned None")

    def properties():
        return [
            {
                "record_id": molecule.GetProp("_Name"),
                "atoms": molecule.GetNumAtoms(),
                "bonds": molecule.GetNumBonds(),
                "method": molecule.GetProp("method") if molecule.HasProp("method") else None,
            }
            for molecule in molecules
        ]

    props = run_stage("05_rdkit_property_extraction", properties, stages)
    record_ids = [row["record_id"] for row in props]
    arrays = run_stage(
        "06_numpy_array_creation",
        lambda: [np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64) for molecule in molecules],
        stages,
    )
    frame = run_stage(
        "07_pandas_dataframe_creation",
        lambda: pd.DataFrame({**{key: [row[key] for row in props] for key in props[0]}, "coordinate_checksum": [float(x.sum()) for x in arrays]}),
        stages,
    )
    existing = run_stage("08_read_existing_per_record", lambda: pd.read_parquet(args.records).iloc[: args.n].copy(), stages)
    merged = run_stage(
        "09_concat_merge",
        lambda: existing.merge(frame, on="record_id", how="inner", validate="one_to_one", suffixes=("", "_probe")),
        stages,
    )
    parquet_path = scratch / "probe.parquet"
    csv_path = scratch / "probe.csv"
    json_path = scratch / "probe.json"
    run_stage("10_parquet_write", lambda: merged.to_parquet(parquet_path, index=False), stages)
    run_stage("11_csv_write", lambda: merged.to_csv(csv_path, index=False), stages)
    run_stage("12_json_write", lambda: json_path.write_text(json.dumps(merged.to_dict("records"), default=str), encoding="utf-8"), stages)

    def cleanup():
        nonlocal supplier, first, molecules, props, arrays, frame, existing, merged
        supplier = first = molecules = props = arrays = frame = existing = merged = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_stage("13_process_cleanup", cleanup, stages)
    output = {
        "schema_version": "sixs-factorial-post-sdf-core-probe-v1",
        "python": sys.version,
        "executable": sys.executable,
        "n": args.n,
        "sdf": str(args.sdf),
        "sdf_sha256": sha256(args.sdf),
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode()).hexdigest(),
        "versions": versions,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_tensor_created": any(
            (stage.get("after", {}).get("torch_cuda_allocated_bytes") or 0) > 0 for stage in stages.values()
        ),
        "stages": stages,
        "outputs": {"parquet_sha256": sha256(parquet_path), "csv_sha256": sha256(csv_path), "json_sha256": sha256(json_path)},
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

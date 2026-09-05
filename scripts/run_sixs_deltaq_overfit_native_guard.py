"""Run the frozen DeltaQ tensor path without importing evaluator-only RDKit.

The Conda RDKit build loads LLVM OpenMP while the PyTorch wheel loads Intel
OpenMP.  The DeltaQ overfit consumes already-materialized ``GraphGeometry``
tensors and does not call RDKit graph construction or SDF evaluation.  This
guard supplies the three historical tensor data helpers used by the frozen
runner and fail-closed stubs for the unused RDKit construction hooks.  It does
not tolerate duplicate OpenMP runtimes or alter scientific calculations.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


def _forbidden_rdkit_path(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("RDKit/evaluator path is forbidden in DeltaQ tensor-only training")


def _install_tensor_only_shims(root: Path) -> None:
    # Importing etflow.ecir normally executes its broad package __init__, which
    # eagerly imports dataset/datamol/RDKit even though the frozen tensor batch
    # does not use them.  Namespace shells preserve ordinary submodule loading
    # from disk without executing those unrelated package initializers.
    etflow_package = types.ModuleType("etflow")
    etflow_package.__path__ = [str(root / "etflow")]
    etflow_package.__package__ = "etflow"
    sys.modules["etflow"] = etflow_package
    ecir_package = types.ModuleType("etflow.ecir")
    ecir_package.__path__ = [str(root / "etflow" / "ecir")]
    ecir_package.__package__ = "etflow.ecir"
    sys.modules["etflow.ecir"] = ecir_package

    rdkit = types.ModuleType("rdkit")
    rdkit.Chem = types.ModuleType("rdkit.Chem")
    sys.modules["rdkit"] = rdkit
    sys.modules["rdkit.Chem"] = rdkit.Chem

    rdkit_utils = types.ModuleType("etflow.ecir.rdkit_utils")
    rdkit_utils.chiral_center_quads = _forbidden_rdkit_path
    sys.modules["etflow.ecir.rdkit_utils"] = rdkit_utils
    target_building = types.ModuleType("etflow.ecir.target_building")
    target_building._record_to_rdkit_mapping = _forbidden_rdkit_path
    sys.modules["etflow.ecir.target_building"] = target_building

    frozen = types.ModuleType("scripts.run_sixs_musigma_reliability_factorial")

    def deltaq_config() -> dict[str, Any]:
        path = root / "configs" / "sixs_deltaq_prototype.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        data = deltaq_config()["data"]
        for key, expected in (
            ("prepared_payload", data["prepared_sha256"]),
            ("source_payload", data["source_payload_sha256"]),
            ("train_manifest", data["train_manifest_sha256"]),
            ("val_manifest", data["val_manifest_sha256"]),
        ):
            if sha256(data[key]) != expected:
                raise RuntimeError(f"frozen input hash changed: {key}")
        prepared = torch.load(data["prepared_payload"], map_location="cpu", weights_only=False)
        sources = torch.load(data["source_payload"], map_location="cpu", weights_only=False)
        if len(prepared["train"]) != 50000 or len(prepared["val"]) != 5000:
            raise RuntimeError("prepared molecule denominator changed")
        if len(sources["train"]) != 150000 or len(sources["val"]) != 10000:
            raise RuntimeError("source-record denominator changed")
        return prepared, sources

    def source_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(str(row["molecule_id"]), []).append(dict(row))
        for values in result.values():
            values.sort(key=lambda value: str(value["sample_id"]))
        return result

    def sample_batch(items, sources, generator, batch, device):
        import torch
        from scripts.run_mcvr_lsgo import collate_graphs

        indices = torch.randint(len(items), (batch,), generator=generator).tolist()
        graphs, source_values, references = [], [], []
        for index in indices:
            item = items[index]
            pool = sources[str(item["molecule_id"])]
            source_i = int(torch.randint(len(pool), (1,), generator=generator))
            ref_i = int(torch.randint(len(item["references"]), (1,), generator=generator))
            graphs.append(item["graph"])
            source_values.append(torch.as_tensor(pool[source_i]["source"], dtype=torch.float64))
            references.append(torch.as_tensor(item["references"][ref_i], dtype=torch.float64))
        return (
            graphs,
            collate_graphs(graphs).to(device),
            torch.cat(source_values).to(device),
            torch.cat(references).to(device),
            indices,
        )

    frozen.load_inputs = load_inputs
    frozen.source_index = source_index
    frozen.sample_batch = sample_batch
    sys.modules["scripts.run_sixs_musigma_reliability_factorial"] = frozen


def _install_blas_free_reporting_correlation() -> None:
    """Avoid initializing a second OpenMP runtime in report-only correlation."""
    import numpy as np

    def scalar_pair_correlation(left, right):
        x = np.asarray(left, dtype=np.float64).reshape(-1)
        y = np.asarray(right, dtype=np.float64).reshape(-1)
        if x.size != y.size or x.size == 0:
            raise ValueError("correlation inputs must be nonempty and equally sized")
        x_centered = x - float(x.mean())
        y_centered = y - float(y.mean())
        denominator = float(
            np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
        )
        value = float(np.sum(x_centered * y_centered) / denominator) if denominator else float("nan")
        return np.asarray([[1.0, value], [value, 1.0]], dtype=np.float64)

    np.corrcoef = scalar_pair_correlation


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: native_guard.py ENTRYPOINT [ARGS ...]")
    entrypoint = Path(sys.argv[1]).resolve()
    _install_tensor_only_shims(entrypoint.parents[1])
    _install_blas_free_reporting_correlation()
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()

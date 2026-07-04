"""Minimal end-to-end sampling smoke test for a 4D Jacobian checkpoint."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Batch

from utils import instantiate_model, read_yaml

from etflow.data import EuclideanDataset

torch.set_float32_matmul_precision("high")


def _shape(value: Any):
    return tuple(value.shape) if hasattr(value, "shape") else None


def _result_keys(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [str(key) for key in value]
    return []


def _save_output(
    output_path: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    torch.save(
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "num_successes": len(records),
            "num_failures": len(failures),
            "molecules": records,
            "failures": failures,
        },
        output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one conformer for a few test molecules with a 4D checkpoint."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_molecules", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--allow_non_jacobian",
        action="store_true",
        help="allow baseline configs with use_jacobian_4d_correction=false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_molecules <= 0:
        raise ValueError("--num_molecules must be positive")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke_output.pt"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; pass --device cpu to use CPU."
        )

    print(f"config: {config_path}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(f"device: {device}", flush=True)

    config = read_yaml(str(config_path))
    model_args = config["model_args"]
    use_jacobian = bool(model_args.get("use_jacobian_4d_correction", False))
    correction_scale = float(model_args.get("jacobian_4d_correction_scale", 0.0))
    print(f"use_jacobian_4d_correction: {use_jacobian}", flush=True)
    print(f"jacobian_4d_correction_scale: {correction_scale}", flush=True)
    if not use_jacobian and not args.allow_non_jacobian:
        raise ValueError("Resolved config has use_jacobian_4d_correction=false")

    # Match scripts/eval.py: same test dataset construction and model loading.
    dataset = EuclideanDataset(
        partition=config["datamodule_args"]["partition"],
        split="test",
    )
    model = instantiate_model(config["model"], model_args)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    jacobian_head = getattr(model.network, "jacobian_4d_head", None)
    if use_jacobian and jacobian_head is None:
        raise RuntimeError("Loaded model has no jacobian_4d_head")

    sampler_args = dict(config.get("eval_args", {}).get("sampler_args", {}))
    print(f"sampler_args: {sampler_args}", flush=True)
    sampling_output = "v_final" if use_jacobian else "v_atom (baseline)"
    print(
        "sampling path: model.sample -> BaseFlow.forward -> "
        f"TorchMDDynamics.forward -> {sampling_output}",
        flush=True,
    )

    stop_idx = min(args.start_idx + args.num_molecules, len(dataset))
    if args.start_idx >= len(dataset):
        raise IndexError(
            f"--start_idx {args.start_idx} is outside test dataset of size {len(dataset)}"
        )

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for idx in range(args.start_idx, stop_idx):
        print("\n" + "=" * 80, flush=True)
        print(f"molecule index: {idx}", flush=True)
        head_calls: Optional[int] = 0 if use_jacobian else None

        def count_head_call(_module, _inputs, _output):
            nonlocal head_calls
            assert head_calls is not None
            head_calls += 1

        hook = (
            jacobian_head.register_forward_hook(count_head_call)
            if jacobian_head is not None
            else None
        )
        try:
            data = dataset[idx]
            smiles = data.smiles
            num_atoms = int(data.atomic_numbers.numel())
            has_rotatable = "rotatable_bond_index" in data
            has_influence = "atom_bond_influence_index" in data
            rotatable = getattr(data, "rotatable_bond_index", None)
            influence = getattr(data, "atom_bond_influence_index", None)

            print(f"smiles: {smiles}", flush=True)
            print(f"num_atoms: {num_atoms}", flush=True)
            print(f"has rotatable_bond_index: {has_rotatable}", flush=True)
            print(f"rotatable_bond_index shape: {_shape(rotatable)}", flush=True)
            print(f"has atom_bond_influence_index: {has_influence}", flush=True)
            print(
                f"atom_bond_influence_index shape: {_shape(influence)}",
                flush=True,
            )
            if rotatable is None or influence is None:
                raise KeyError(
                    "Dataset item is missing rotatable/influence tensors required by 4D sampling"
                )

            # Match eval.py's Batch and model.sample call. A single copy produces
            # one conformer and isolates the forward path from full evaluation cost.
            batched_data = Batch.from_data_list([data])
            z = batched_data["atomic_numbers"].to(device)
            edge_index = batched_data["edge_index"].to(device)
            batch = batched_data["batch"].to(device)
            node_attr = batched_data["node_attr"].to(device)
            chiral_index = batched_data["chiral_index"].to(device)
            chiral_nbr_index = batched_data["chiral_nbr_index"].to(device)
            chiral_tag = batched_data["chiral_tag"].to(device)
            batched_rotatable = batched_data["rotatable_bond_index"].to(device)
            batched_influence = batched_data["atom_bond_influence_index"].to(device)

            start_time = time.perf_counter()
            with torch.no_grad():
                result = model.sample(
                    z,
                    edge_index,
                    batch,
                    node_attr=node_attr,
                    chiral_index=chiral_index,
                    chiral_nbr_index=chiral_nbr_index,
                    chiral_tag=chiral_tag,
                    rotatable_bond_index=batched_rotatable,
                    atom_bond_influence_index=batched_influence,
                    **sampler_args,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - start_time

            print(f"sampling result type: {type(result).__name__}", flush=True)
            print(f"sampling result shape: {_shape(result)}", flush=True)
            print(f"sampling result keys: {_result_keys(result)}", flush=True)
            print("saved result key: generated_pos", flush=True)
            print(f"jacobian_4d_head forward calls: {head_calls}", flush=True)
            print(f"elapsed_seconds: {elapsed_seconds:.3f}", flush=True)
            if use_jacobian and head_calls == 0:
                raise RuntimeError(
                    "Sampling completed without entering jacobian_4d_head"
                )
            if not torch.is_tensor(result):
                raise TypeError(
                    f"model.sample returned {type(result).__name__}, expected Tensor"
                )
            if tuple(result.shape) != (num_atoms, 3):
                raise ValueError(
                    f"Unexpected sample shape {tuple(result.shape)}; "
                    f"expected {(num_atoms, 3)}"
                )
            if not torch.isfinite(result).all():
                raise ValueError("Sampling result contains NaN or Inf")

            records.append(
                {
                    "index": idx,
                    "smiles": smiles,
                    "num_atoms": num_atoms,
                    "rotatable_bond_index_shape": tuple(rotatable.shape),
                    "atom_bond_influence_index_shape": tuple(influence.shape),
                    "jacobian_4d_head_calls": head_calls,
                    "elapsed_seconds": elapsed_seconds,
                    "generated_pos": result.detach().cpu(),
                    "reference_pos": data.pos.detach().cpu(),
                }
            )
            _save_output(
                output_path,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                records=records,
                failures=failures,
            )
            print(f"saved: {output_path}", flush=True)
        except Exception as exc:
            failure = {
                "index": idx,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"molecule {idx} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            _save_output(
                output_path,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                records=records,
                failures=failures,
            )
        finally:
            if hook is not None:
                hook.remove()
            if args.debug and device.type == "cuda":
                print(
                    f"cuda_memory_allocated: {torch.cuda.memory_allocated(device)}",
                    flush=True,
                )

    print("\n" + "=" * 80, flush=True)
    print(f"attempted: {stop_idx - args.start_idx}", flush=True)
    print(f"succeeded: {len(records)}", flush=True)
    print(f"failed: {len(failures)}", flush=True)
    print(f"output: {output_path}", flush=True)
    if records:
        print("SMOKE PASSED: at least one molecule generated and was saved", flush=True)
        return 0
    print("SMOKE FAILED: all attempted molecules failed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

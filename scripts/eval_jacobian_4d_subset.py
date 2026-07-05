"""Generate an incrementally saved subset in the formal ETFlow eval format."""

from __future__ import annotations

import argparse
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from utils import instantiate_model, read_yaml

from etflow.commons import save_pkl
from etflow.commons.featurization import get_sample_field
from etflow.data import EuclideanDataset

torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--data_dir",
        help="Processed-data root containing drugs/{train,val,test}; defaults to DATA_DIR/processed.",
    )
    parser.add_argument("--num_molecules", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_non_jacobian", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    return parser.parse_args()


def _save_outputs(
    output_dir: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    model_type: str,
    compatible_records: List[Data],
    diagnostic_records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    times: List[float],
    split: str,
    start_idx: int,
    requested_molecules: int,
    seed: int,
) -> None:
    # This is the same list-of-Data format consumed by eval_cov_mat.py.
    save_pkl(output_dir / "generated_files.pkl", compatible_records)
    save_pkl(output_dir / "times.pkl", times)
    torch.save(
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "model_type": model_type,
            "split": split,
            "start_idx": start_idx,
            "requested_molecules": requested_molecules,
            "sample_seed": seed,
            "num_successes": len(diagnostic_records),
            "num_failures": len(failures),
            "molecules": diagnostic_records,
            "failures": failures,
            "sampling_times_per_conformer": times,
            "formal_eval_file": str(output_dir / "generated_files.pkl"),
        },
        output_dir / "subset_output.pt",
    )


def _reference_positions(dataset: EuclideanDataset, idx: int) -> torch.Tensor:
    raw_sample = torch.load(
        dataset.data_files[idx], map_location="cpu", weights_only=False
    )
    pos_ref = get_sample_field(raw_sample, "pos")
    if pos_ref is None:
        raise ValueError(f"Raw sample {dataset.data_files[idx]} has no pos field")
    pos_ref = torch.as_tensor(pos_ref).cpu()
    if pos_ref.ndim == 2:
        pos_ref = pos_ref.unsqueeze(0)
    if pos_ref.ndim != 3 or pos_ref.size(-1) != 3 or pos_ref.size(0) == 0:
        raise ValueError(
            f"Reference positions must have shape [C, N, 3], got {tuple(pos_ref.shape)}"
        )
    return pos_ref


def main() -> int:
    args = parse_args()
    if args.num_molecules <= 0:
        raise ValueError("--num_molecules must be positive")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative")
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    config = read_yaml(str(config_path))
    model_args = config["model_args"]
    use_jacobian = bool(model_args.get("use_jacobian_4d_correction", False))
    model_type = "4D" if use_jacobian else "base"
    if not use_jacobian and not args.allow_non_jacobian:
        raise ValueError(
            "Resolved config has use_jacobian_4d_correction=false; "
            "pass --allow_non_jacobian for a baseline run"
        )

    print(f"config: {config_path}", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"model_type: {model_type}", flush=True)
    print(f"split: {args.split}", flush=True)
    print(f"sample_seed: {args.seed}", flush=True)
    print(f"use_jacobian_4d_correction: {use_jacobian}", flush=True)

    dataset = EuclideanDataset(
        partition=config["datamodule_args"]["partition"],
        split=args.split,
        data_dir=args.data_dir,
    )
    if args.start_idx >= len(dataset):
        raise IndexError(
            f"--start_idx {args.start_idx} outside {args.split} dataset of size {len(dataset)}"
        )
    stop_idx = min(args.start_idx + args.num_molecules, len(dataset))

    model = instantiate_model(config["model"], model_args)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    jacobian_head = getattr(model.network, "jacobian_4d_head", None)
    if use_jacobian and jacobian_head is None:
        raise RuntimeError("4D config loaded a model without jacobian_4d_head")

    eval_args = config.get("eval_args", {})
    max_batch_size = int(eval_args.get("batch_size", 32))
    sampler_args = dict(eval_args.get("sampler_args", {}))
    if max_batch_size <= 0:
        raise ValueError("eval_args.batch_size must be positive")
    print(f"max_batch_size: {max_batch_size}", flush=True)
    print(f"sampler_args: {sampler_args}", flush=True)

    compatible_records: List[Data] = []
    diagnostic_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    times: List[float] = []

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
            pos_ref = _reference_positions(dataset, idx)
            smiles = data.smiles
            num_atoms = int(data.atomic_numbers.numel())
            if pos_ref.size(1) != num_atoms:
                raise ValueError(
                    f"Reference atom count {pos_ref.size(1)} != graph atoms {num_atoms}"
                )
            rotatable = data.rotatable_bond_index
            influence = data.atom_bond_influence_index
            num_samples = 2 * int(pos_ref.size(0))
            pos_chunks = []
            molecule_times = []
            print(f"smiles: {smiles}", flush=True)
            print(f"num_atoms: {num_atoms}", flush=True)
            print(f"num_reference_conformers: {pos_ref.size(0)}", flush=True)
            print(f"num_generated_conformers: {num_samples}", flush=True)
            print(f"rotatable_bond_index shape: {tuple(rotatable.shape)}", flush=True)
            print(
                f"atom_bond_influence_index shape: {tuple(influence.shape)}",
                flush=True,
            )

            for batch_start in range(0, num_samples, max_batch_size):
                batch_size = min(max_batch_size, num_samples - batch_start)
                batched_data = Batch.from_data_list([data] * batch_size)
                z = batched_data["atomic_numbers"].to(device)
                edge_index = batched_data["edge_index"].to(device)
                batch = batched_data["batch"].to(device)
                node_attr = batched_data["node_attr"].to(device)
                chiral_index = batched_data["chiral_index"].to(device)
                chiral_nbr_index = batched_data["chiral_nbr_index"].to(device)
                chiral_tag = batched_data["chiral_tag"].to(device)
                batched_rotatable = batched_data["rotatable_bond_index"].to(device)
                batched_influence = batched_data[
                    "atom_bond_influence_index"
                ].to(device)

                started = time.perf_counter()
                with torch.no_grad():
                    generated = model.sample(
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
                seconds_per_conformer = (
                    time.perf_counter() - started
                ) / batch_size
                molecule_times.extend([seconds_per_conformer] * batch_size)
                generated = generated.view(batch_size, num_atoms, 3).detach().cpu()
                if not torch.isfinite(generated).all():
                    raise ValueError("Generated positions contain NaN or Inf")
                pos_chunks.append(generated)

            pos_gen = torch.cat(pos_chunks, dim=0)
            if tuple(pos_gen.shape) != (num_samples, num_atoms, 3):
                raise ValueError(f"Unexpected generated shape {tuple(pos_gen.shape)}")
            if use_jacobian and head_calls == 0:
                raise RuntimeError("Sampling did not enter jacobian_4d_head")

            times.extend(molecule_times)
            compatible_records.append(
                Data(
                    mol_id=Path(dataset.data_files[idx]).stem,
                    source_mol_id=Path(dataset.data_files[idx]).stem,
                    dataset_index=idx,
                    split=args.split,
                    smiles=smiles,
                    atomic_numbers=data.atomic_numbers.cpu(),
                    pos_ref=pos_ref.numpy(),
                    rdmol=data.mol,
                    pos_gen=pos_gen.numpy(),
                )
            )
            diagnostic_records.append(
                {
                    "index": idx,
                    "smiles": smiles,
                    "num_atoms": num_atoms,
                    "model_type": model_type,
                    "jacobian_4d_head_calls": head_calls,
                    "sampling_time_mean": float(np.mean(molecule_times)),
                    "generated_pos": pos_gen,
                    "reference_pos": pos_ref,
                }
            )
            print(f"generated_pos shape: {tuple(pos_gen.shape)}", flush=True)
            print(f"jacobian_4d_head_calls: {head_calls}", flush=True)
            print(
                f"sampling_time_mean: {np.mean(molecule_times):.3f}s",
                flush=True,
            )
            if len(diagnostic_records) % args.save_every == 0:
                _save_outputs(
                    output_dir,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    compatible_records=compatible_records,
                    diagnostic_records=diagnostic_records,
                    failures=failures,
                    times=times,
                    split=args.split,
                    start_idx=args.start_idx,
                    requested_molecules=args.num_molecules,
                    seed=args.seed,
                )
                print(f"incremental save: {output_dir}", flush=True)
        except Exception as exc:
            failures.append(
                {
                    "index": idx,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"molecule {idx} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            _save_outputs(
                output_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                compatible_records=compatible_records,
                diagnostic_records=diagnostic_records,
                failures=failures,
                times=times,
                split=args.split,
                start_idx=args.start_idx,
                requested_molecules=args.num_molecules,
                seed=args.seed,
            )
        finally:
            if hook is not None:
                hook.remove()
            if args.debug and device.type == "cuda":
                print(
                    f"cuda_memory_allocated: {torch.cuda.memory_allocated(device)}",
                    flush=True,
                )

    _save_outputs(
        output_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        compatible_records=compatible_records,
        diagnostic_records=diagnostic_records,
        failures=failures,
        times=times,
        split=args.split,
        start_idx=args.start_idx,
        requested_molecules=args.num_molecules,
        seed=args.seed,
    )
    print("\n" + "=" * 80, flush=True)
    print(f"num_successes: {len(diagnostic_records)}", flush=True)
    print(f"num_failures: {len(failures)}", flush=True)
    print(f"subset_output: {output_dir / 'subset_output.pt'}", flush=True)
    print(f"formal_eval_file: {output_dir / 'generated_files.pkl'}", flush=True)
    requested = stop_idx - args.start_idx
    if len(diagnostic_records) == requested and not failures:
        print("SUBSET SAMPLING PASSED", flush=True)
        return 0
    print(
        f"SUBSET SAMPLING FAILED: requested={requested}, "
        f"successes={len(diagnostic_records)}, failures={len(failures)}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

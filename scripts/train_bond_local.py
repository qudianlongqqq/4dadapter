"""Training entry point for ETFlow bond-local velocity regularization."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger as log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import instantiate_model, read_yaml

from etflow.data.datamodule import BaseDataModule

torch.set_float32_matmul_precision("high")


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _resolve_config(args) -> Dict[str, Any]:
    config = read_yaml(args.config)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {args.config}")

    datamodule_args = config.setdefault("datamodule_args", {})
    dataloader_args = datamodule_args.setdefault("dataloader_args", {})
    model_args = config.setdefault("model_args", {})
    trainer_args = dict(config.get("trainer_args", {}))

    if args.batch_size is not None:
        dataloader_args["batch_size"] = _positive_int(args.batch_size, "batch_size")
    if args.bond_velocity_loss_weight is not None:
        model_args["bond_velocity_loss_weight"] = float(
            args.bond_velocity_loss_weight
        )
    if args.bond_velocity_on_rotatable_only is not None:
        model_args["bond_velocity_on_rotatable_only"] = bool(
            args.bond_velocity_on_rotatable_only
        )

    model_args.setdefault("use_bond_local_velocity_loss", True)
    model_args.setdefault("bond_velocity_loss_weight", 0.003)
    model_args.setdefault("bond_velocity_on_rotatable_only", False)

    if not model_args["use_bond_local_velocity_loss"]:
        raise ValueError(
            "train_bond_local.py requires use_bond_local_velocity_loss=true."
        )
    if (
        not math.isfinite(model_args["bond_velocity_loss_weight"])
        or model_args["bond_velocity_loss_weight"] < 0
    ):
        raise ValueError(
            "model_args.bond_velocity_loss_weight must be finite and non-negative, got "
            f"{model_args['bond_velocity_loss_weight']}."
        )
    if model_args.get("use_angular_head", False) or model_args.get(
        "use_angular_loss", False
    ):
        raise ValueError(
            "train_bond_local.py requires the angular head and angular loss to be disabled."
        )

    trainer_args.update(
        {
            "max_steps": _positive_int(args.max_steps, "max_steps"),
            "accumulate_grad_batches": _positive_int(
                args.accumulate_grad_batches,
                "accumulate_grad_batches",
            ),
            "val_check_interval": _positive_int(
                args.val_check_interval,
                "val_check_interval",
            ),
            "limit_val_batches": _positive_int(
                args.limit_val_batches,
                "limit_val_batches",
            ),
            "log_every_n_steps": _positive_int(
                args.log_every_n_steps,
                "log_every_n_steps",
            ),
            "enable_checkpointing": True,
            "default_root_dir": str(args.output_dir),
        }
    )
    trainer_args.pop("fast_dev_run", None)
    trainer_args.pop("logger", None)
    trainer_args.pop("callbacks", None)

    config["seed"] = int(args.seed)
    config["trainer"] = "Trainer"
    config["trainer_args"] = trainer_args
    config["logger"] = "CSVLogger"
    config["logger_args"] = {
        "save_dir": str(args.output_dir),
        "name": "csv_logs",
    }
    config["callbacks"] = [
        {
            "callback": "ModelCheckpoint",
            "callback_args": {
                "dirpath": str(args.output_dir / "checkpoints"),
                "save_top_k": 3,
                "save_last": True,
                "monitor": "val/loss",
                "mode": "min",
            },
        }
    ]
    config["pretrained_ckpt"] = (
        args.pretrained_ckpt
        if args.pretrained_ckpt is not None
        else config.get("pretrained_ckpt")
    )
    config["output_dir"] = str(args.output_dir)
    return config


def _load_pretrained_weights(model, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        log.info("No pretrained checkpoint requested; training from initialized weights.")
        return

    expanded_path = Path(
        os.path.expandvars(os.path.expanduser(checkpoint_path))
    ).resolve()
    if not expanded_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {expanded_path}")

    checkpoint = torch.load(expanded_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint does not contain a state dict: {expanded_path}"
        )

    model_keys = set(model.state_dict())
    matching_keys = model_keys.intersection(state_dict)
    if not matching_keys:
        raise RuntimeError(
            "Pretrained checkpoint has no parameter names matching this model. "
            f"Checkpoint: {expanded_path}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded pretrained model weights only from: {expanded_path}")
    log.info(f"Matched parameter keys: {len(matching_keys)}/{len(model_keys)}")
    log.info(f"Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys}")
    log.info(
        f"Unexpected keys ({len(incompatible.unexpected_keys)}): "
        f"{incompatible.unexpected_keys}"
    )


def _save_resolved_config(config: Dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "config.resolved.yaml"
    with output_path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return output_path


def train(args) -> None:
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    args.config = str(config_path)

    config = _resolve_config(args)
    resolved_path = _save_resolved_config(config, args.output_dir)
    datamodule_args = config["datamodule_args"]
    trainer_args = config["trainer_args"]

    log.info(f"config path: {config_path}")
    log.info(f"resolved config: {resolved_path}")
    log.info(f"output_dir: {args.output_dir}")
    log.info(
        f"batch_size: {datamodule_args['dataloader_args'].get('batch_size')}"
    )
    log.info(
        "bond_velocity_loss_weight: "
        f"{config['model_args']['bond_velocity_loss_weight']}"
    )
    log.info(
        "bond_velocity_on_rotatable_only: "
        f"{config['model_args']['bond_velocity_on_rotatable_only']}"
    )
    log.info(f"max_steps: {trainer_args['max_steps']}")
    log.info(
        f"accumulate_grad_batches: {trainer_args['accumulate_grad_batches']}"
    )
    log.info(
        "skip_unrecoverable_mol: "
        f"{datamodule_args.get('skip_unrecoverable_mol', True)}"
    )
    log.info(f"bad_sample_csv: {datamodule_args.get('bad_sample_csv')}")

    seed_everything(config["seed"], workers=True)
    datamodule = BaseDataModule(**datamodule_args)
    model = instantiate_model(config["model"], config["model_args"])
    _load_pretrained_weights(model, config.get("pretrained_ckpt"))

    csv_logger = CSVLogger(
        save_dir=str(args.output_dir),
        name="csv_logs",
    )
    csv_logger.log_hyperparams(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="bond-local-{step}",
        save_top_k=3,
        save_last=True,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        **trainer_args,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)

    log.info(f"best checkpoint path: {checkpoint_callback.best_model_path}")
    log.info(f"last checkpoint path: {checkpoint_callback.last_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ETFlow with bond-local velocity consistency regularization."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--bond_velocity_loss_weight", type=float, default=None)
    parser.add_argument(
        "--bond_velocity_on_rotatable_only",
        action="store_true",
        default=None,
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--val_check_interval", type=int, default=500)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

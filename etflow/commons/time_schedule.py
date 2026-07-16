"""Explicit inference-time schedules shared by Cartesian and ECIR refiners."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
from torch import Tensor


TIME_SCHEDULE_MODES = ("legacy_full", "train_range", "fixed", "explicit")


class InferenceTimeRangeWarning(RuntimeWarning):
    """Inference requests times outside the checkpoint training range."""


def inference_time_schedule(
    reference: Tensor,
    steps: int,
    *,
    mode: str = "train_range",
    training_t_min: float = 0.0,
    training_t_max: float = 1.0,
    inference_t_min: float | None = None,
    inference_t_max: float | None = None,
    fixed_t: float | None = None,
    explicit_time_schedule: Sequence[float] | Tensor | None = None,
    strict_training_range: bool = False,
) -> Tensor:
    """Return a fully specified schedule; one step selects its lower endpoint."""

    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    if mode not in TIME_SCHEDULE_MODES:
        raise ValueError(f"Unknown time_schedule_mode={mode!r}; expected {TIME_SCHEDULE_MODES}")
    training_t_min = float(training_t_min)
    training_t_max = float(training_t_max)
    if training_t_min > training_t_max:
        raise ValueError("training_t_min must not exceed training_t_max")
    if mode == "legacy_full":
        values = torch.linspace(0.0, 1.0, steps, device=reference.device, dtype=reference.dtype)
    elif mode == "train_range":
        lower = training_t_min if inference_t_min is None else float(inference_t_min)
        upper = training_t_max if inference_t_max is None else float(inference_t_max)
        if lower > upper:
            raise ValueError("inference_t_min must not exceed inference_t_max")
        values = torch.linspace(lower, upper, steps, device=reference.device, dtype=reference.dtype)
    elif mode == "fixed":
        if fixed_t is None:
            raise ValueError("fixed time schedule requires fixed_t")
        values = reference.new_full((steps,), float(fixed_t))
    else:
        if explicit_time_schedule is None:
            raise ValueError("explicit time schedule requires explicit_time_schedule")
        values = torch.as_tensor(
            explicit_time_schedule, device=reference.device, dtype=reference.dtype
        ).reshape(-1)
        if values.numel() != steps:
            raise ValueError(
                f"explicit_time_schedule has {values.numel()} values for steps={steps}"
            )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("inference time schedule contains NaN or Inf")
    outside = bool(
        (values < training_t_min - 1.0e-8).any()
        or (values > training_t_max + 1.0e-8).any()
    )
    if outside:
        message = (
            f"Inference time range [{float(values.min()):.6g}, {float(values.max()):.6g}] "
            f"exceeds checkpoint training range [{training_t_min:.6g}, {training_t_max:.6g}]"
        )
        if strict_training_range:
            raise ValueError(message)
        warnings.warn(message, InferenceTimeRangeWarning, stacklevel=2)
    return values

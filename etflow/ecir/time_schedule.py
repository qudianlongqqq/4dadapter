"""Backward-compatible ECIR import path for the shared time scheduler."""

from etflow.commons.time_schedule import (
    InferenceTimeRangeWarning,
    TIME_SCHEDULE_MODES,
    inference_time_schedule,
)

__all__ = [
    "InferenceTimeRangeWarning",
    "TIME_SCHEDULE_MODES",
    "inference_time_schedule",
]

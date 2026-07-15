"""Serial Cartesian-to-Global4D residual refinement components."""

from .cache import (
    SERIAL_CACHE_SCHEMA_VERSION,
    SerialGlobal4DResidualDataset,
    build_stage2_training_record,
    label_free_cartesian_view,
    load_frozen_cartesian_teacher,
    rollout_frozen_cartesian,
    validate_stage2_inference_record,
    validate_stage2_training_record,
)
from .oracle import (
    benefit_aware_gate_target,
    solve_serial_residual_oracle,
)
from .model import SerialGlobal4DResidualRefiner
from .targets import materialize_stage2_targets
from .safety import (
    SafeUpdateResult,
    geometry_guard,
    safe_serial_update,
    trust_region_clip,
)

__all__ = [
    "SERIAL_CACHE_SCHEMA_VERSION",
    "SerialGlobal4DResidualDataset",
    "SerialGlobal4DResidualRefiner",
    "SafeUpdateResult",
    "benefit_aware_gate_target",
    "build_stage2_training_record",
    "label_free_cartesian_view",
    "load_frozen_cartesian_teacher",
    "materialize_stage2_targets",
    "geometry_guard",
    "rollout_frozen_cartesian",
    "solve_serial_residual_oracle",
    "safe_serial_update",
    "trust_region_clip",
    "validate_stage2_inference_record",
    "validate_stage2_training_record",
]

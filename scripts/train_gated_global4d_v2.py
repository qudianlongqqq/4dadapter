#!/usr/bin/env python
"""Train Gated Global4D V2 without any strict-mode default fallback."""

from __future__ import annotations

try:
    from train_global_coupled_4d_flow import main
except ModuleNotFoundError:
    from scripts.train_global_coupled_4d_flow import main


if __name__ == "__main__":
    main(
        default_config="configs/gated_global4d_v2_pilot.yaml",
        required_fusion_mode="gated_additive",
    )

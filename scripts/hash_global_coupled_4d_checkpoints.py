#!/usr/bin/env python
"""Find inference-identical checkpoints without trusting their filenames."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    checkpoint_inference_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    identities = {}
    canonical_by_hash = {}
    reuse_plan = {}
    for name in args.names:
        identity = checkpoint_inference_identity(args.checkpoint_dir / f"{name}.ckpt")
        digest = identity["inference_sha256"]
        canonical = canonical_by_hash.setdefault(digest, name)
        identities[name] = identity
        reuse_plan[name] = canonical
    payload = {
        "identities": identities,
        "canonical_checkpoint": reuse_plan,
        "duplicates": {
            name: canonical
            for name, canonical in reuse_plan.items()
            if name != canonical
        },
    }
    atomic_json_save(payload, args.output)
    for name, canonical in reuse_plan.items():
        print(f"{name}: {'unique' if name == canonical else f'reused_from={canonical}'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible alias. The only maintained pipeline is the one-command
# Smoke 200 -> independently initialized Formal 5k runner.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
exec bash scripts/run_global_coupled_4d_smoke_and_matched.sh "$@"

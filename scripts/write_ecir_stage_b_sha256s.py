#!/usr/bin/env python
"""Write a deterministic SHA256 manifest for completed Stage B artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "ecir_mvr" / "STAGE_B_SHA256SUMS.txt"
ARTIFACTS = (
    "diagnostics/ecir_mvr/stage_b/all_results.csv",
    "diagnostics/ecir_mvr/stage_b/pareto_front.csv",
    "diagnostics/ecir_mvr/stage_b/source_summary.csv",
    "diagnostics/ecir_mvr/stage_b/severity_summary.csv",
    "diagnostics/ecir_mvr/stage_b/decision.json",
    "data/ecir_mvr/validity_reference_stats.json",
    "data/ecir_mvr/stage_b_views/manifest.parquet",
    "data/ecir_mvr/stage_b_views/metadata.json",
    "docs/MCVR_CHEMICAL_VALIDITY_DEFINITION.md",
    "docs/MCVR_STAGE_B_REPORT.md",
    "reports/ecir_mvr/progressive_state.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    paths = [ROOT / relative for relative in ARTIFACTS]
    paths.extend(sorted((ROOT / "data/ecir_mvr/stage_b_views/coordinates").glob("*.pt")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Stage B artifacts: {missing}")
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()

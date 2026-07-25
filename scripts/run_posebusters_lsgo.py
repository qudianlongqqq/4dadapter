#!/usr/bin/env python3
"""Run the frozen, paired PoseBusters evaluation for the LSGO pilot."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from posebusters import PoseBusters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/learned_geometry"
CONFIG = Path(r"E:\miniconda\envs\external-validity\Lib\site-packages\posebusters\config\mol_fast.yml")


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def selected_columns(config: dict) -> list[str]:
    result: list[str] = []
    for module in config.get("modules", []):
        rename = module.get("rename_outputs", {})
        for raw in module.get("chosen_binary_test_output", []):
            name = str(rename.get(raw, raw)).lower().replace(" ", "_")
            if name not in result:
                result.append(name)
    return result


def main() -> int:
    freeze_path = OUT / "COORDINATE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or not freeze.get("external_evaluation_unlocked"):
        raise RuntimeError("PoseBusters remains locked")
    if freeze.get("source_plus_condition_count") != 5 or set(freeze.get("stopped_conditions", {})) != {"C-G", "C-P"}:
        raise RuntimeError("coordinate-freeze scope changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    checks = selected_columns(config)
    if len(checks) < 10:
        raise RuntimeError(f"unexpected PoseBusters schema: {checks}")
    evaluator = PoseBusters(config=config, max_workers=4, chunk_size=50)
    summaries: list[dict] = []
    bindings: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}
    started = time.time()
    for export in freeze["exports"]:
        method = str(export["method"])
        sdf_path = Path(export["sdf_path"])
        if sha256(sdf_path) != export["sdf_sha256"]:
            raise RuntimeError(f"frozen SDF changed: {method}")
        target = OUT / f"per_record/posebusters/{method}__primary.parquet"
        if target.is_file():
            frame = pd.read_parquet(target)
        else:
            raw = evaluator.bust(sdf_path, None, None, full_report=True).reset_index()
            missing = [column for column in checks if column not in raw.columns]
            if missing:
                raise RuntimeError(f"PoseBusters columns missing: {missing}")
            frame = pd.DataFrame({"sample_id": raw["molecule"].astype(str)})
            for column in checks:
                frame[column] = raw[column].fillna(False).astype(bool)
            frame["pb_overall"] = frame[checks].all(axis=1)
            frame["method"] = method
            frame["candidate"] = "primary"
            atomic_frame(target, frame)
        required = {"sample_id", "method", "candidate", "pb_overall", *checks}
        if len(frame) != 600 or frame.sample_id.astype(str).nunique() != 600 or not required.issubset(frame.columns):
            raise RuntimeError(f"incomplete PoseBusters result: {method}")
        frames[method] = frame
        row = {
            "method": method, "variant": export.get("variant"), "seed": export.get("seed"),
            "records": 600, "overall": float(frame.pb_overall.mean()),
            **{column: float(frame[column].mean()) for column in checks},
        }
        summaries.append(row)
        bindings.append({
            "method": method, "sdf_sha256": export["sdf_sha256"],
            "result_path": str(target), "result_sha256": sha256(target),
            "sample_identity_sha256": __import__("hashlib").sha256(
                "\n".join(frame.sample_id.astype(str)).encode()
            ).hexdigest(),
        })
        print(json.dumps({"method": method, "overall": row["overall"]}), flush=True)

    source = frames["Source"].set_index("sample_id")
    transitions: list[dict] = []
    for method, frame in frames.items():
        if method == "Source":
            continue
        paired = frame.set_index("sample_id").loc[source.index]
        source_pass = source.pb_overall.astype(bool)
        output_pass = paired.pb_overall.astype(bool)
        transitions.append({
            "method": method, "records": 600,
            "pass_to_fail": int((source_pass & ~output_pass).sum()),
            "pass_to_fail_fraction": float((source_pass & ~output_pass).mean()),
            "fail_to_pass": int((~source_pass & output_pass).sum()),
            "fail_to_pass_fraction": float((~source_pass & output_pass).mean()),
            "strict_failure_transfer": int((source_pass & ~output_pass).sum()),
        })
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(tables / "POSEBUSTERS_SUMMARY.csv", index=False)
    pd.DataFrame(transitions).to_csv(tables / "POSEBUSTERS_TRANSITIONS.csv", index=False)
    payload = {
        "schema_version": "mcvr-lsgo-posebusters-v1", "status": "COMPLETED",
        "version": "0.6.5", "config_path": str(CONFIG), "config_sha256": sha256(CONFIG),
        "coordinate_freeze_sha256": sha256(freeze_path), "checks": checks,
        "summaries": summaries, "transitions": transitions, "bindings": bindings,
        "runtime_seconds": time.time() - started,
        "formal_test_records_read": 0, "frozen_holdout_records_read": 0,
        "full10k_used_for_tuning": False, "used_for_model_selection": False,
    }
    atomic_json(OUT / "manifests/POSEBUSTERS_COMPLETE.json", payload)
    print("LSGO_POSEBUSTERS_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

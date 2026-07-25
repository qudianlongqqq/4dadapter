#!/usr/bin/env python3
"""Paired frozen-coordinate PoseBusters evaluation for BAT."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from posebusters import PoseBusters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ecir_mvr/bat_refinement"
CONFIG = Path(r"E:\miniconda\envs\external-validity\Lib\site-packages\posebusters\config\mol_fast.yml")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)


def atomic_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def selected_columns(config):
    result = []
    for module in config.get("modules", []):
        rename = module.get("rename_outputs", {})
        for raw in module.get("chosen_binary_test_output", []):
            name = str(rename.get(raw, raw)).lower().replace(" ", "_")
            if name not in result: result.append(name)
    return result


def main() -> int:
    freeze_path = OUT / "COORDINATE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = {"Source", "BA_seed173", "BA+C_seed173", "BA_seed181", "BA+C_seed181", "BA_seed193", "BA+C_seed193"}
    if freeze.get("status") != "FROZEN" or not freeze.get("external_evaluation_unlocked") or set(freeze.get("conditions", [])) != expected:
        raise RuntimeError("PoseBusters remains locked or coordinate scope changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); checks = selected_columns(config)
    if len(checks) < 10: raise RuntimeError("unexpected PoseBusters schema")
    evaluator = PoseBusters(config=config, max_workers=4, chunk_size=50)
    summaries, bindings, frames = [], [], {}; started = time.time()
    for export in freeze["exports"]:
        method = export["method"]; sdf_path = Path(export["sdf_path"])
        if sha256(sdf_path) != export["sdf_sha256"]: raise RuntimeError(f"frozen SDF changed: {method}")
        target = OUT / f"per_record/posebusters/{method}__primary.parquet"
        if target.is_file():
            frame = pd.read_parquet(target)
        else:
            raw = evaluator.bust(sdf_path, None, None, full_report=True).reset_index()
            missing = [column for column in checks if column not in raw.columns]
            if missing: raise RuntimeError(f"PoseBusters columns missing: {missing}")
            frame = pd.DataFrame({"sample_id": raw["molecule"].astype(str)})
            for column in checks: frame[column] = raw[column].fillna(False).astype(bool)
            frame["pb_overall"] = frame[checks].all(axis=1); frame["method"] = method; frame["candidate"] = "primary"
            atomic_frame(target, frame)
        if len(frame) != 600 or frame.sample_id.nunique() != 600: raise RuntimeError(f"incomplete PB: {method}")
        frames[method] = frame
        row = {"method": method, "seed": export.get("seed"), "records": 600, "overall": float(frame.pb_overall.mean()), **{column: float(frame[column].mean()) for column in checks}}
        summaries.append(row); bindings.append({"method": method, "sdf_sha256": export["sdf_sha256"], "result_path": str(target), "result_sha256": sha256(target)})
        print(json.dumps({"method": method, "overall": row["overall"]}), flush=True)
    source = frames["Source"].set_index("sample_id"); transitions = []
    for method, frame in frames.items():
        if method == "Source": continue
        paired = frame.set_index("sample_id").loc[source.index]
        overall_source, overall_output = source.pb_overall.astype(bool), paired.pb_overall.astype(bool)
        row = {"method": method, "pass_to_fail": int((overall_source & ~overall_output).sum()), "fail_to_pass": int((~overall_source & overall_output).sum())}
        for check in checks:
            before, after = source[check].astype(bool), paired[check].astype(bool)
            row[f"{check}__pass_to_fail"] = int((before & ~after).sum()); row[f"{check}__fail_to_pass"] = int((~before & after).sum())
        transitions.append(row)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT / "tables/POSEBUSTERS_SUMMARY.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "tables/POSEBUSTERS_TRANSITIONS.csv", index=False)
    payload = {"schema_version": "mcvr-bat-posebusters-v1", "status": "COMPLETED", "version": "0.6.5", "config_path": str(CONFIG), "config_sha256": sha256(CONFIG), "coordinate_freeze_sha256": sha256(freeze_path), "checks": checks, "summaries": summaries, "transitions": transitions, "bindings": bindings, "runtime_seconds": time.time() - started, "formal_test_records_read": 0, "frozen_holdout_records_read": 0, "used_for_model_selection": False}
    atomic_json(OUT / "manifests/POSEBUSTERS_COMPLETE.json", payload)
    print("BAT_POSEBUSTERS_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

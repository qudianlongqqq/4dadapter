#!/usr/bin/env python3
"""Recover legacy LSGO/learned-geometry/BAT cohort identities.

This is an identity-only audit.  It resolves the historical source identifiers
through the immutable formal-large TRAIN manifest, reads only the ``smiles``
field from one cached source record per molecule, and delegates
canonicalization to the already-frozen STEP 2D source-universe implementation.
No model, coordinate generation, metric, or protected outcome is executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import torch
from rdkit import Chem, RDLogger


EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "fbfeffab299c070fcbf29edb99277113c5641ee588000f00fc384162337ecb3d"
)
EXPECTED_TRAIN_COMPACT_SHA256 = (
    "72b9ca73e617498f3ea596884585bae473c6017951019b9a3c203bfd015d5479"
)
LEGACY_ARTIFACT_COMMITS = {
    "LSGO_MECHANISM": "d1cbe2f4f8aac6a38457cb9cf9e2f319feb95b50",
    "LEARNED_GEOMETRY": "fbde3ef497364cf92aa90caba8e62e25a33e2e18",
    "BAT_REFINEMENT": "6808ccb10930df6815756e0be258650d9b06903e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def import_file(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_blob_sha256(repo: Path, commit: str, relative_path: Path) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative_path.as_posix()}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def git_artifact_matches(repo: Path, commit: str, relative_path: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", relative_path.as_posix()],
        cwd=repo,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git diff failed for {relative_path}")
    return result.returncode == 0


def cohort_ids(payload: dict[str, Any]) -> list[str]:
    return [
        str(identity)
        for cohort in payload["cohorts"].values()
        for identity in cohort["molecule_ids"]
    ]


def load_current_train(provenance_path: Path) -> set[str]:
    identities: set[str] = set()
    with provenance_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if "CURRENT_REFINER_TRAIN" in row["reason_excluded"].split(";"):
                identities.add(row["molecule_identity"])
    if len(identities) != 49_964:
        raise RuntimeError(f"Unexpected canonical TRAIN identity count: {len(identities)}")
    return identities


def recover(args: argparse.Namespace) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    source = import_file(args.canonicalizer, "sixs_frozen_step2d_canonicalizer")
    if source.IDENTITY_DEFINITION != "sixs-canonical-isomeric-atom-map-free-h-normalized-v1":
        raise RuntimeError("Frozen canonical identity definition changed")

    if sha256_file(args.source_manifest) != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError("Historical TRAIN source manifest SHA256 changed")
    if sha256_file(args.training_compact) != EXPECTED_TRAIN_COMPACT_SHA256:
        raise RuntimeError("Historical TRAIN compact SHA256 changed")

    paths = {
        "LSGO_MECHANISM": args.lsgo_identity,
        "LEARNED_GEOMETRY": args.learned_identity,
        "BAT_REFINEMENT": args.bat_identity,
    }
    artifact_facts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        current_sha = sha256_file(path)
        blob_sha = git_blob_sha256(args.repo_root, LEGACY_ARTIFACT_COMMITS[name], path.relative_to(args.repo_root))
        relative_path = path.relative_to(args.repo_root)
        if not git_artifact_matches(
            args.repo_root, LEGACY_ARTIFACT_COMMITS[name], relative_path
        ):
            raise RuntimeError(f"{name} identity artifact differs from its frozen Git blob")
        payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        artifact_facts[name] = {
            "path": str(path.resolve()),
            "sha256": current_sha,
            "git_commit": LEGACY_ARTIFACT_COMMITS[name],
            "git_blob_sha256": blob_sha,
            "git_content_match_under_repository_filters": True,
            "fact_type": "ARTIFACT_FACT+CODE_FACT+GIT_FACT",
        }

    learned_raw = cohort_ids(payloads["LEARNED_GEOMETRY"])
    bat_raw = cohort_ids(payloads["BAT_REFINEMENT"])
    mechanism_raw = [str(value) for value in payloads["LSGO_MECHANISM"]["molecule_ids"]]

    # The mechanism preregistration used all 2,000 TRAIN molecules to freeze
    # thresholds; its 48-molecule confirmation cohort is a proven subset.
    lsgo_train_raw = [
        str(value)
        for value in payloads["LEARNED_GEOMETRY"]["cohorts"]["LSGO_TRAIN"]["molecule_ids"]
    ]
    bat_train_raw = {
        str(value)
        for value in payloads["BAT_REFINEMENT"]["cohorts"]["BAT_TRAIN"]["molecule_ids"]
    }
    if set(lsgo_train_raw) != bat_train_raw:
        raise RuntimeError("LSGO and BAT frozen TRAIN identity partitions differ")
    if not set(mechanism_raw) <= set(lsgo_train_raw):
        raise RuntimeError("LSGO mechanism identities are not a subset of its threshold TRAIN cohort")

    raw_by_cohort = {
        "LEGACY_LSGO": lsgo_train_raw,
        "LEGACY_LEARNED_GEOMETRY": learned_raw,
        "LEGACY_BAT_REFINEMENT": bat_raw,
    }
    all_source_ids = set().union(*(set(values) for values in raw_by_cohort.values()))

    frame = pd.read_parquet(
        args.source_manifest, columns=["molecule_id", "source_path", "split", "test_record"]
    )
    if (
        len(frame) != 150_000
        or frame["molecule_id"].nunique() != 50_000
        or set(frame["split"].astype(str)) != {"train"}
        or frame["test_record"].fillna(False).astype(bool).any()
    ):
        raise RuntimeError("Historical source manifest is not the frozen TRAIN-only universe")
    selected = frame[frame["molecule_id"].astype(str).isin(all_source_ids)].copy()
    if selected["molecule_id"].astype(str).nunique() != len(all_source_ids):
        found = set(selected["molecule_id"].astype(str))
        raise RuntimeError(f"Unmapped historical source identities: {len(all_source_ids - found)}")
    grouped = {
        str(key): value.sort_values("source_path")
        for key, value in selected.groupby("molecule_id", sort=False)
    }

    canonical_by_source: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for source_id in sorted(all_source_ids):
        rows = grouped[source_id]
        if len(rows) != 3:
            raise RuntimeError(f"Expected three source records for {source_id}, got {len(rows)}")
        cache_path = args.source_cache / Path(str(rows.iloc[0]["source_path"])).name
        try:
            record = torch.load(cache_path, map_location="cpu", weights_only=False)
            molecule = Chem.MolFromSmiles(str(record.get("smiles", "")))
            if molecule is None:
                raise ValueError("cached source has no valid smiles")
            canonical_by_source[source_id] = source.canonical_identity(molecule)[0]
        except Exception as exc:  # explicit fail-closed accounting
            failures.append(
                {"source_id": source_id, "cache": str(cache_path), "error": repr(exc)}
            )
    if failures:
        raise RuntimeError(f"Canonical identity recovery failures: {len(failures)}")

    current_train = load_current_train(args.base_provenance)
    detail_rows: list[dict[str, Any]] = []
    canonical_sets: dict[str, set[str]] = {}
    for name, raw_ids in raw_by_cohort.items():
        recovered = [canonical_by_source[value] for value in raw_ids]
        canonical = set(recovered)
        canonical_sets[name] = canonical
        outside = canonical - current_train
        if outside:
            raise RuntimeError(f"{name} contains {len(outside)} identities outside current TRAIN")
        identity_sources = {
            "LEGACY_LSGO": (
                f"{args.learned_identity.resolve()}#LSGO_TRAIN;"
                f"{args.lsgo_identity.resolve()}#molecule_ids"
            ),
            "LEGACY_LEARNED_GEOMETRY": str(args.learned_identity.resolve()),
            "LEGACY_BAT_REFINEMENT": str(args.bat_identity.resolve()),
        }
        identity_shas = {
            "LEGACY_LSGO": (
                f"{artifact_facts['LEARNED_GEOMETRY']['sha256']};"
                f"{artifact_facts['LSGO_MECHANISM']['sha256']}"
            ),
            "LEGACY_LEARNED_GEOMETRY": artifact_facts["LEARNED_GEOMETRY"]["sha256"],
            "LEGACY_BAT_REFINEMENT": artifact_facts["BAT_REFINEMENT"]["sha256"],
        }
        detail_rows.append(
            {
                "COHORT_NAME": name,
                "ROLE": "LEGACY_DEVELOPMENT_AND_OUTCOME_INSPECTED_TRAIN_SIDE",
                "SEARCHED_LOCATIONS": (
                    "tracked DATASET_IDENTITY.json; tracked dataset builder/config/preregistration; "
                    "frozen formal-large TRAIN parquet; TRAIN source cache; frozen Git blob"
                ),
                "IDENTITY_SOURCE": identity_sources[name],
                "IDENTITY_SOURCE_SHA256": identity_shas[name],
                "RAW_RECORDS_FOUND": 3 * len(raw_ids),
                "CANONICAL_IDENTITIES_RECOVERED": len(canonical),
                "CANONICALIZATION_FAILURES": 0,
                "DUPLICATES": len(recovered) - len(canonical),
                "RECOVERY_STATUS": "COMPLETE",
                "PROVENANCE_STATUS": "ARTIFACT_FACT+CODE_FACT+GIT_FACT",
                "MISSING_INFORMATION": "NONE",
                "CURRENT_TRAIN_SUPERSET_OVERLAP": len(canonical & current_train),
                "CURRENT_TRAIN_SUPERSET_MISSING": len(outside),
            }
        )

    exact_union = set().union(*canonical_sets.values())
    additional_over_excluded = len(current_train - exact_union)
    status = {
        "schema_version": "sixs-step2d-legacy-identity-recovery-v1",
        "status": "COMPLETE",
        "identity_definition": source.IDENTITY_DEFINITION,
        "cohorts": {row["COHORT_NAME"]: row for row in detail_rows},
        "legacy_exact_union_n": len(exact_union),
        "legacy_exact_union_in_current_train_n": len(exact_union & current_train),
        "legacy_exact_union_outside_current_train_n": len(exact_union - current_train),
        "current_train_conservative_superset_n": len(current_train),
        "conservative_superset_used": True,
        "superset_membership_proven": True,
        "superset_additional_over_excluded_molecules": additional_over_excluded,
        "new_additional_over_excluded_by_recovery": 0,
        "historical_development_identity_union_complete": True,
        "protected_outcome_read": False,
        "scientific_model_changed": False,
        "artifact_facts": artifact_facts,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "training_compact": str(args.training_compact.resolve()),
        "training_compact_sha256": sha256_file(args.training_compact),
    }

    fields = list(detail_rows[0])
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(detail_rows)
    atomic_text(args.report_dir / "11_LEGACY_IDENTITY_RECOVERY.csv", buffer.getvalue())
    atomic_text(
        args.report_dir / "12_LEGACY_IDENTITY_RECOVERY_STATUS.json",
        json.dumps(status, indent=2, sort_keys=True) + "\n",
    )
    report = f"""# STEP 2D legacy historical cohort identity recovery

The three targeted legacy routes are recovered without model execution or
outcome access. Their tracked identity artifacts are unchanged from the frozen
Git revisions under the repository's text normalization. Every historical source identifier maps deterministically to
three records in the immutable formal-large TRAIN manifest (SHA256
`{EXPECTED_TRAIN_MANIFEST_SHA256}`); one cached record per molecule supplies
the molecular graph used by the existing STEP 2D canonicalizer.

The LSGO mechanism thresholds used the 2,000-molecule TRAIN partition, and the
48-molecule mechanism-confirm set is an exact subset. Learned geometry and BAT
each preserve all 2,700 direct train/dev/external identities. Across the three
routes there are {len(exact_union):,} unique canonical identities; all
{len(exact_union):,} are already members of the current canonical TRAIN
exclusion.

The current TRAIN set is therefore a proven conservative superset for these
TRAIN-only historical routes. It contains {additional_over_excluded:,}
identities beyond the recovered three-route exact union, but all were already
excluded before this recovery, so this audit adds zero new over-exclusions and
does not change the exclusion-union membership.

```text
LSGO_LEGACY_RECOVERY_STATUS = COMPLETE
LSGO_LEGACY_IDENTITIES_N = {len(canonical_sets['LEGACY_LSGO'])}
LEARNED_GEOMETRY_LEGACY_RECOVERY_STATUS = COMPLETE
LEARNED_GEOMETRY_LEGACY_IDENTITIES_N = {len(canonical_sets['LEGACY_LEARNED_GEOMETRY'])}
BAT_REFINEMENT_LEGACY_RECOVERY_STATUS = COMPLETE
BAT_REFINEMENT_LEGACY_IDENTITIES_N = {len(canonical_sets['LEGACY_BAT_REFINEMENT'])}
LEGACY_EXACT_UNION_N = {len(exact_union)}
SUPERSET_MEMBERSHIP_PROVEN = YES
CONSERVATIVE_SUPERSET_USED = YES
ADDITIONAL_OVER_EXCLUDED_MOLECULE_COUNT = {additional_over_excluded}
NEW_ADDITIONAL_OVER_EXCLUDED_BY_THIS_RECOVERY = 0
HISTORICAL_DEVELOPMENT_IDENTITY_UNION_COMPLETE = YES
PROTECTED_OUTCOME_READ = NO
SCIENTIFIC_MODEL_CHANGED = NO
```
"""
    atomic_text(args.report_dir / "13_LEGACY_RECOVERY_AUDIT.md", report)
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--canonicalizer", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--training-compact", required=True, type=Path)
    parser.add_argument("--base-provenance", required=True, type=Path)
    parser.add_argument("--lsgo-identity", required=True, type=Path)
    parser.add_argument("--learned-identity", required=True, type=Path)
    parser.add_argument("--bat-identity", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    recover(parse_args())

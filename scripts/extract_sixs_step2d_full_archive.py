#!/usr/bin/env python3
"""Reliably materialize the official Torsional Diffusion DRUGS archive on Windows.

The upstream archive uses Linux filenames derived from SMILES.  A literal
backslash is a valid Linux filename character but a Windows path separator, so
raw ``DRUGS/drugs/*.pickle`` members are stored under deterministic numeric
names and accompanied by a lossless JSONL member map.  All other members retain
their archive-relative path.  Existing verified identity assets may be reused
through hard links, avoiding a second physical copy of the large standardized
pickle chunks.

This utility only extracts source data.  It performs no model inference or
scientific evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import BinaryIO


EXPECTED_ARCHIVE_BYTES = 25_308_038_362
EXPECTED_ARCHIVE_SHA256 = (
    "79812ffefcb51abc5ceca04ce1b67d1af385e1ffe7ffb3c74a5e9fa2d4bb69cd"
)
RAW_PREFIX = "DRUGS/drugs/"
RAW_SUFFIX = ".pickle"
STATUS_NAME = "FULL_EXTRACTION_STATUS.json"
MEMBER_MAP_NAME = "RAW_ARCHIVE_MEMBER_MAP.jsonl"


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


def normalized_member_name(name: str) -> str:
    normalized = name
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path.as_posix()


def destination_for(name: str, raw_ordinal: int) -> tuple[Path, str | None]:
    """Return a Windows-safe relative path and optional original raw stem."""
    if name.startswith(RAW_PREFIX) and name.endswith(RAW_SUFFIX):
        source_stem = name[len(RAW_PREFIX) : -len(RAW_SUFFIX)]
        return (
            Path("DRUGS") / "drugs_windows_safe" / f"{raw_ordinal:06d}.pickle",
            source_stem,
        )
    parts = PurePosixPath(name).parts
    if any("\\" in part or ":" in part or "*" in part or "?" in part for part in parts):
        raise ValueError(f"Unsupported Windows archive filename outside raw DRUGS: {name!r}")
    return Path(*parts), None


def copy_atomic(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        shutil.copyfileobj(source, stream, length=16 * 1024 * 1024)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def try_hardlink(source: Path, destination: Path, expected_size: int) -> bool:
    if not source.is_file() or source.stat().st_size != expected_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".linkpart")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError:
        if temporary.exists():
            temporary.unlink()
        return False
    return True


def extract(
    archive: Path,
    output: Path,
    reuse_root: Path | None,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, object]:
    archive = archive.resolve()
    output = output.resolve()
    if archive.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Archive byte-size mismatch: {archive.stat().st_size} != {expected_bytes}"
        )
    archive_sha256 = sha256_file(archive)
    if archive_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Archive SHA256 mismatch: {archive_sha256} != {expected_sha256}"
        )

    status_path = output / STATUS_NAME
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("status") == "COMPLETE"
            and status.get("archive_sha256") == archive_sha256
            and status.get("archive_bytes") == expected_bytes
        ):
            return status
        raise RuntimeError(f"Existing extraction marker is incompatible: {status_path}")

    output.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    member_count = 0
    file_count = 0
    directory_count = 0
    raw_count = 0
    total_file_bytes = 0
    materialized_bytes = 0
    hardlinked_files = 0
    hardlinked_bytes = 0
    reused_files = 0
    reused_bytes = 0

    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            member_count += 1
            name = normalized_member_name(member.name)
            if member.isdir():
                directory_count += 1
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Unsupported non-regular archive member {name!r} type={member.type!r}"
                )

            if name.startswith(RAW_PREFIX) and name.endswith(RAW_SUFFIX):
                raw_count += 1
            relative, raw_stem = destination_for(name, raw_count)
            destination = output / relative
            file_count += 1
            total_file_bytes += member.size

            if destination.is_file() and destination.stat().st_size == member.size:
                reused_files += 1
                reused_bytes += member.size
            else:
                linked = False
                if reuse_root is not None and raw_stem is None:
                    reuse_source = reuse_root.resolve() / relative
                    linked = try_hardlink(reuse_source, destination, member.size)
                if linked:
                    hardlinked_files += 1
                    hardlinked_bytes += member.size
                else:
                    free = shutil.disk_usage(output).free
                    reserve = 1024**3
                    if free < member.size + reserve:
                        raise RuntimeError(
                            "Insufficient free disk space for next archive member: "
                            f"free={free}, member={member.size}, reserve={reserve}, name={name}"
                        )
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Cannot read archive member: {name}")
                    copy_atomic(source, destination)
                    materialized_bytes += member.size

            if raw_stem is not None:
                raw_rows.append(
                    {
                        "raw_ordinal": raw_count,
                        "archive_member": name,
                        "source_stem": raw_stem,
                        "windows_safe_relative_path": relative.as_posix(),
                        "bytes": member.size,
                    }
                )

    map_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in raw_rows
    )
    atomic_text(output / MEMBER_MAP_NAME, map_text)

    required = [
        output / "DRUGS" / "split.npy",
        output / "DRUGS" / "test_mols.pkl",
        output / "DRUGS" / "test_smiles.csv",
        output / "DRUGS" / "standardized_pickles",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Required extracted assets missing: {missing}")
    standardized = sorted((required[-1]).glob("*.pickle"))

    status: dict[str, object] = {
        "schema_version": "sixs-step2d-full-archive-extraction-v1",
        "status": "COMPLETE",
        "archive": str(archive),
        "archive_bytes": expected_bytes,
        "archive_sha256": archive_sha256,
        "output_root": str(output),
        "archive_members": member_count,
        "archive_regular_files": file_count,
        "archive_directories": directory_count,
        "archive_regular_file_bytes": total_file_bytes,
        "materialized_new_bytes": materialized_bytes,
        "hardlinked_files": hardlinked_files,
        "hardlinked_bytes": hardlinked_bytes,
        "resumed_existing_files": reused_files,
        "resumed_existing_bytes": reused_bytes,
        "raw_pickle_members": raw_count,
        "raw_member_storage": "WINDOWS_SAFE_NUMERIC_NAMES_WITH_LOSSLESS_JSONL_MAP",
        "raw_member_map": str(output / MEMBER_MAP_NAME),
        "raw_member_map_sha256": sha256_file(output / MEMBER_MAP_NAME),
        "standardized_pickle_chunks": len(standardized),
        "split_boltz_10k_present": (
            output / "DRUGS" / "split_boltz_10k.npy"
        ).is_file(),
    }
    atomic_text(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--expected-bytes", type=int, default=EXPECTED_ARCHIVE_BYTES)
    parser.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = extract(
        args.archive,
        args.output,
        args.reuse_root,
        expected_bytes=args.expected_bytes,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stream-extract only identity/reference assets from official drugs.tar.gz."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(name: str) -> bool:
    path = PurePosixPath(name)
    if path.as_posix() in {
        "DRUGS/split.npy",
        "DRUGS/split_boltz_10k.npy",
        "DRUGS/test_mols.pkl",
        "DRUGS/test_smiles.csv",
    }:
        return True
    return (
        len(path.parts) == 3
        and path.parts[:2] == ("DRUGS", "standardized_pickles")
        and path.suffix == ".pickle"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    marker = output / "IDENTITY_EXTRACTION_STATUS.json"
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed extraction: {marker}")
    output.mkdir(parents=True, exist_ok=True)

    members: list[str] = []
    raw_names: list[str] = []
    extracted: list[str] = []
    with tarfile.open(archive, mode="r|gz") as tar:
        for index, member in enumerate(tar, start=1):
            name = PurePosixPath(member.name).as_posix().lstrip("./")
            members.append(name)
            parts = PurePosixPath(name).parts
            if (
                member.isfile()
                and len(parts) == 3
                and parts[:2] == ("DRUGS", "drugs")
                and PurePosixPath(name).suffix == ".pickle"
            ):
                raw_names.append(PurePosixPath(name).name)
            if member.isfile() and wanted(name):
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot extract {name}")
                destination = output.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
                os.replace(temporary, destination)
                extracted.append(name)
            if index % 25_000 == 0:
                print(
                    f"ARCHIVE_SCAN members={index} raw_names={len(raw_names)} "
                    f"extracted={len(extracted)}",
                    flush=True,
                )

    raw_names.sort()
    atomic_text(output / "ARCHIVE_MEMBERS.txt", "\n".join(members) + "\n")
    atomic_text(output / "RAW_PICKLE_FILENAMES.txt", "\n".join(raw_names) + "\n")
    split = output / "DRUGS" / "split.npy"
    standard = sorted((output / "DRUGS" / "standardized_pickles").glob("*.pickle"))
    status = {
        "schema_version": "sixs-step2d-torsional-asset-extraction-v1",
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_members": len(members),
        "raw_pickle_filenames": len(raw_names),
        "raw_pickle_filename_index_sha256": sha256_file(
            output / "RAW_PICKLE_FILENAMES.txt"
        ),
        "standardized_pickle_files": len(standard),
        "split_sha256": sha256_file(split),
        "extracted_files": len(extracted),
        "status": "COMPLETE",
    }
    atomic_text(marker, json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

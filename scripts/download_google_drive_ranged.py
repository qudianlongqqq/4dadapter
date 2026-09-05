#!/usr/bin/env python3
"""Resume a public Google Drive file with independent HTTP byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import threading
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.stat().st_size if output.exists() else 0
    if existing > args.size:
        raise RuntimeError(f"Existing file is larger than expected: {existing} > {args.size}")
    if existing == args.size:
        print(f"DOWNLOAD_ALREADY_COMPLETE bytes={existing}", flush=True)
        return

    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={args.file_id}&export=download&confirm=t"
    )
    remaining = args.size - existing
    segment = (remaining + args.workers - 1) // args.workers
    ranges: list[tuple[int, int]] = []
    for worker in range(args.workers):
        start = existing + worker * segment
        end = min(args.size - 1, start + segment - 1)
        if start <= end:
            ranges.append((start, end))

    with output.open("r+b" if output.exists() else "w+b") as stream:
        stream.truncate(args.size)

    lock = threading.Lock()
    completed = existing
    last_report = existing
    started = time.monotonic()
    chunk_size = args.chunk_mib * 1024 * 1024

    def fetch(byte_range: tuple[int, int]) -> int:
        nonlocal completed, last_report
        start, end = byte_range
        current = start
        session = requests.Session()
        while current <= end:
            response = session.get(
                url,
                headers={"Range": f"bytes={current}-{end}"},
                stream=True,
                timeout=(30, 180),
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"Range request {current}-{end} returned {response.status_code}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {current}-"):
                raise RuntimeError(
                    f"Unexpected Content-Range for {current}-{end}: {content_range!r}"
                )
            try:
                with output.open("r+b", buffering=0) as stream:
                    stream.seek(current)
                    for block in response.iter_content(chunk_size=chunk_size):
                        if not block:
                            continue
                        if current + len(block) - 1 > end:
                            block = block[: end - current + 1]
                        stream.write(block)
                        current += len(block)
                        with lock:
                            completed += len(block)
                            if completed - last_report >= 1024 * 1024 * 1024:
                                elapsed = max(time.monotonic() - started, 1e-9)
                                rate = (completed - existing) / elapsed / (1024 * 1024)
                                print(
                                    f"DOWNLOAD_PROGRESS bytes={completed} "
                                    f"pct={100.0 * completed / args.size:.2f} "
                                    f"session_MiB_s={rate:.2f}",
                                    flush=True,
                                )
                                last_report = completed
            except (requests.RequestException, OSError):
                time.sleep(2)
                continue
        return end - start + 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        received = sum(executor.map(fetch, ranges))
    if received != remaining or completed != args.size or output.stat().st_size != args.size:
        raise RuntimeError(
            f"Incomplete download: received={received}, completed={completed}, "
            f"size={output.stat().st_size}, expected={args.size}"
        )
    print(f"DOWNLOAD_COMPLETE bytes={args.size}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Keep the unattended heartbeat fresh while validation blocks the train loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.run_timing import RunTiming, write_heartbeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=45.0)
    args = parser.parse_args()
    timing = RunTiming(args.output_dir)
    (args.output_dir / "heartbeat_monitor.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    timing.mark("heartbeat_monitor_start", pid=os.getpid(), interval_seconds=args.interval)
    while True:
        heartbeat_path = args.output_dir / "heartbeat.json"
        if heartbeat_path.is_file():
            heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            if heartbeat.get("status") not in {"RUNNING", None}:
                break
            write_heartbeat(args.output_dir)
        time.sleep(args.interval)
    timing.mark("heartbeat_monitor_end", status=heartbeat.get("status"))


if __name__ == "__main__":
    main()

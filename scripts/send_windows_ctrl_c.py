#!/usr/bin/env python
"""Deliver a normal console Ctrl+C event to an existing Windows process."""

from __future__ import annotations

import argparse
import ctypes
import os
import time


CTRL_C_EVENT = 0
ATTACH_PARENT_PROCESS = -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("console Ctrl+C delivery is implemented only for Windows")
    kernel32 = ctypes.windll.kernel32
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(args.pid):
        error = ctypes.get_last_error()
        raise RuntimeError(f"AttachConsole({args.pid}) failed with Windows error {error}")
    try:
        if not kernel32.SetConsoleCtrlHandler(None, True):
            raise RuntimeError("failed to make the sender ignore its own Ctrl+C event")
        if not kernel32.GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0):
            error = ctypes.get_last_error()
            raise RuntimeError(f"GenerateConsoleCtrlEvent failed with Windows error {error}")
        time.sleep(max(args.settle_seconds, 0.0))
    finally:
        kernel32.FreeConsole()


if __name__ == "__main__":
    main()

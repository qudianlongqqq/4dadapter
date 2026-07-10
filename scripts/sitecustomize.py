"""Automatic repository bootstrap for every directly executed script.

Python imports ``sitecustomize`` during startup from the script directory, so
legacy entry points also gain the repository root without requiring PYTHONPATH.
New entry points additionally call ``_bootstrap.bootstrap`` explicitly.
"""

from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
Thin wrapper so `python main.py ...` works from a cloned repo without
installing. The actual CLI lives at eliminability.cli.main for the
`pip install` entry point.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eliminability.cli import main

if __name__ == "__main__":
    sys.exit(main())

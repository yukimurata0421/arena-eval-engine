#!/usr/bin/env python3
"""
Deprecated wrapper. Use `arena run ...` instead.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from arena.cli import main

if __name__ == "__main__":
    print("[WARN] Direct execution of scripts/master.py is deprecated. Use `arena run`.\n")
    sys.exit(main(["run", *sys.argv[1:]]))

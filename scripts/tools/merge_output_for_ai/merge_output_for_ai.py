# -*- coding: utf-8 -*-
"""
Compatibility entrypoint for the artifact export subsystem.

The original CLI path is preserved and delegated to
`scripts.tools.artifacts.cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.tools.artifacts.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

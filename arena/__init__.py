"""Repository-local import shim for the public release layer.

This package ensures `import arena` resolves to this checkout's `src/arena`
even when another editable install exists on the same machine.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ARENA = _REPO_ROOT / "src" / "arena"

if not _SRC_ARENA.exists():
    raise ImportError(f"missing local arena source tree: {_SRC_ARENA}")

# Keep this package importable and prioritize src/arena for submodules.
__path__ = [str(_SRC_ARENA), str(Path(__file__).resolve().parent)]

__all__ = ["__version__"]
__version__ = "0.2.9"

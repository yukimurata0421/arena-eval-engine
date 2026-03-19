from __future__ import annotations

from pathlib import Path

import arena
import arena.artifacts


def _assert_under_repo(module_file: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = Path(module_file).resolve()
    assert repo_root in module_path.parents, f"module imported from outside repo: {module_path}"


def test_arena_imports_resolve_to_arena_release_tree() -> None:
    _assert_under_repo(arena.__file__ or "")
    _assert_under_repo(arena.artifacts.__file__ or "")


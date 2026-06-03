from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_REVIEWABLE_PROSE_LINE = 120
FENCE_RE = re.compile(r"^\s*(```|~~~)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:")


def _markdown_paths() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _reviewable_lines(path: Path) -> list[tuple[int, str]]:
    reviewable: list[tuple[int, str]] = []
    in_fence = False

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue

        stripped = line.lstrip()
        if (
            in_fence
            or not line.strip()
            or stripped.startswith("#")
            or stripped.startswith("|")
            or line.startswith("[![")
            or line.startswith(("    ", "\t"))
            or REFERENCE_LINK_RE.match(line)
        ):
            continue

        reviewable.append((line_number, line))

    return reviewable


def test_markdown_prose_lines_are_raw_reviewable() -> None:
    offenders = []
    for path in _markdown_paths():
        for line_number, line in _reviewable_lines(path):
            if len(line) > MAX_REVIEWABLE_PROSE_LINE:
                relative_path = path.relative_to(ROOT)
                offenders.append(f"{relative_path}:{line_number} len={len(line)}")

    assert not offenders, "Long markdown prose lines:\n" + "\n".join(offenders)

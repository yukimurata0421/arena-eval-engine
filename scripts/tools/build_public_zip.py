from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DEFAULT_INCLUDE = [
    "src",
    "scripts",
    "tests",
    "docker",
    "docs",
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "LICENSE",
]

FORBIDDEN_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "build",
    "dist",
}
FORBIDDEN_FILE_NAMES = {
    ".coverage",
    ".DS_Store",
    "Thumbs.db",
}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_PATH_PREFIXES = (
    ".git/",
    "output/",
    "scripts/tools/debug/tmp_",
)
FORBIDDEN_EXACT_PATHS = {
    "scripts/structure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean public source zip.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    parser.add_argument(
        "--output",
        default="arena_public_source.zip",
        help="Output zip path (absolute or repo-root-relative).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="Files/directories to include (repo-root-relative).",
    )
    return parser.parse_args()


def _is_forbidden(rel_posix: str) -> bool:
    if rel_posix in FORBIDDEN_EXACT_PATHS:
        return True
    if any(rel_posix.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
        return True

    parts = rel_posix.split("/")
    if any(part in FORBIDDEN_DIR_NAMES for part in parts[:-1]):
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True

    name = parts[-1]
    if name in FORBIDDEN_FILE_NAMES:
        return True
    if name.endswith(FORBIDDEN_SUFFIXES):
        return True
    return False


def _iter_files(repo_root: Path, includes: list[str]) -> list[Path]:
    files: list[Path] = []
    for inc in includes:
        p = (repo_root / inc).resolve()
        if not p.exists():
            continue
        if p.is_file():
            rel = p.relative_to(repo_root).as_posix()
            if not _is_forbidden(rel):
                files.append(p)
            continue
        for child in sorted(x for x in p.rglob("*") if x.is_file()):
            rel = child.relative_to(repo_root).as_posix()
            if _is_forbidden(rel):
                continue
            files.append(child)
    return sorted(set(files))


def _write_deterministic_file(zf: ZipFile, abs_path: Path, rel_path: str) -> None:
    data = abs_path.read_bytes()
    info = ZipInfo(rel_path)
    info.date_time = (2020, 1, 1, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data)


def build_zip(repo_root: Path, output_path: Path, includes: list[str]) -> tuple[int, int]:
    files = _iter_files(repo_root, includes)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for abs_path in files:
            rel_path = abs_path.relative_to(repo_root).as_posix()
            _write_deterministic_file(zf, abs_path, rel_path)
            total_bytes += abs_path.stat().st_size

    with ZipFile(output_path, mode="r") as zf:
        bad = [name for name in zf.namelist() if _is_forbidden(name)]
        if bad:
            raise RuntimeError(f"forbidden entries found in zip: {bad[:5]}")

    return len(files), total_bytes


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    try:
        count, size = build_zip(repo_root, output_path, list(args.include))
    except Exception as exc:
        print(f"[ERROR] failed to build zip: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] zip: {output_path}")
    print(f"[OK] files: {count}")
    print(f"[OK] uncompressed bytes: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

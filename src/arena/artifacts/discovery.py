from __future__ import annotations

from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from arena.artifacts.policies import (
    AI_EXPORT_DIR_PREFIX,
    AI_FALLBACK_SOURCE_PATHS,
    AI_FILENAME_GLOB_FALLBACKS,
    AI_PRIORITY_B_ALLOWED_EXTS,
    AI_PRIORITY_B_DISCOVERY_EXCLUDE_GLOBS,
    AI_PRIORITY_B_DISCOVERY_INCLUDE_GLOBS,
    AI_PRIORITY_B_MAX_FILE_BYTES,
    ALWAYS_EXCLUDE_DIRS,
    ALWAYS_EXCLUDE_DIR_PREFIXES,
    ALWAYS_EXCLUDE_REL_PATHS,
    OUTPUT_DIR,
    ROOT,
    check_ai_export_exclusion,
)


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def enumerate_files(base_dir: Path) -> list[Path]:
    return [path for path in base_dir.rglob("*") if path.is_file()]


def normalize_rel(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def resolve_optional_export_source(base_dir: Path, relative_path: str) -> Path | None:
    for path in [base_dir / relative_path, Path(OUTPUT_DIR) / relative_path, ROOT / "output" / relative_path]:
        if path.exists() and path.is_file():
            return path
    return None


def _candidate_key_for_base(base_dir: Path, path: Path) -> tuple[int, int, float, str]:
    rel = path.relative_to(base_dir)
    parent_parts = rel.parts[:-1]
    has_prefixed_parent = any(part.startswith("_") for part in parent_parts)
    depth = len(parent_parts)
    mtime = path.stat().st_mtime
    return (1 if has_prefixed_parent else 0, depth, -mtime, rel.as_posix())


def _iter_ai_search_roots(base_dir: Path) -> list[Path]:
    raw_roots = [base_dir, Path(OUTPUT_DIR), ROOT / "output", ROOT / "scripts"]
    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw_roots:
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        if root.exists() and root.is_dir():
            roots.append(root)
    return roots


def _find_candidates_by_glob(search_root: Path, filename_pattern: str) -> list[Path]:
    return [path for path in search_root.rglob(filename_pattern) if path.is_file()]


def resolve_ai_source_path(base_dir: Path, relative_path: str) -> tuple[Path | None, str]:
    primary = base_dir / Path(relative_path)
    if primary.exists() and primary.is_file():
        return primary, ""

    fallback = AI_FALLBACK_SOURCE_PATHS.get(relative_path)
    if fallback is not None and fallback.exists() and fallback.is_file():
        return fallback, f"fallback_source={fallback}"

    search_roots = _iter_ai_search_roots(base_dir)
    root_rank = {str(root.resolve()): index for index, root in enumerate(search_roots)}

    glob_patterns = AI_FILENAME_GLOB_FALLBACKS.get(relative_path, [])
    if glob_patterns:
        candidates: dict[str, tuple[Path, Path]] = {}
        for search_root in search_roots:
            for pattern in glob_patterns:
                for path in _find_candidates_by_glob(search_root, pattern):
                    candidates[str(path.resolve())] = (search_root, path)

        def glob_sort_key(item: tuple[Path, Path]) -> tuple[int, int, int, float, str]:
            search_root, path = item
            rank = root_rank.get(str(search_root.resolve()), 99)
            prefixed, depth, neg_mtime, rel = _candidate_key_for_base(search_root, path)
            return (rank, prefixed, depth, neg_mtime, rel)

        ranked_candidates = sorted(candidates.values(), key=glob_sort_key)
        if ranked_candidates:
            selected_root, selected = ranked_candidates[0]
            rel_selected = selected.relative_to(selected_root).as_posix()
            return selected, f"matched_by_glob={rel_selected}; root={selected_root}; candidates={len(ranked_candidates)}"

    if "/" not in relative_path and "\\" not in relative_path:
        candidates: dict[str, tuple[Path, Path]] = {}
        for search_root in search_roots:
            for path in search_root.rglob(relative_path):
                if path.is_file():
                    candidates[str(path.resolve())] = (search_root, path)

        def name_sort_key(item: tuple[Path, Path]) -> tuple[int, int, int, float, str]:
            search_root, path = item
            rank = root_rank.get(str(search_root.resolve()), 99)
            prefixed, depth, neg_mtime, rel = _candidate_key_for_base(search_root, path)
            return (rank, prefixed, depth, neg_mtime, rel)

        ranked_candidates = sorted(candidates.values(), key=name_sort_key)
        if ranked_candidates:
            selected_root, selected = ranked_candidates[0]
            rel_selected = selected.relative_to(selected_root).as_posix()
            if len(ranked_candidates) == 1:
                return selected, f"matched_from={rel_selected}; root={selected_root}"
            return selected, f"matched_from={rel_selected}; root={selected_root}; ambiguous_matches={len(ranked_candidates)}"

    return None, "source file not found"


def _iter_ai_discovery_roots(base_dir: Path) -> list[Path]:
    raw_roots = [base_dir, Path(OUTPUT_DIR), ROOT / "output"]
    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw_roots:
        try:
            resolved = str(root.resolve())
        except Exception:
            resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        if root.exists() and root.is_dir():
            roots.append(root)
    return roots


def _is_ai_discovery_excluded_path(relative_path: str) -> bool:
    rel = relative_path.replace("\\", "/").lower()
    parts = Path(rel).parts
    if any(part in ALWAYS_EXCLUDE_DIRS for part in parts):
        return True
    if any(part.startswith(ALWAYS_EXCLUDE_DIR_PREFIXES) for part in parts):
        return True
    if any(part.startswith(AI_EXPORT_DIR_PREFIX.lower()) for part in parts):
        return True
    if rel in ALWAYS_EXCLUDE_REL_PATHS:
        return True
    return any(fnmatch(rel, pattern) for pattern in AI_PRIORITY_B_DISCOVERY_EXCLUDE_GLOBS)


def discover_priority_b_existing_targets(base_dir: Path) -> list[str]:
    discovered: dict[str, str] = {}
    for root in _iter_ai_discovery_roots(base_dir):
        for pattern in AI_PRIORITY_B_DISCOVERY_INCLUDE_GLOBS:
            for path in _find_candidates_by_glob(root, pattern):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(root).as_posix()
                except Exception:
                    continue
                if _is_ai_discovery_excluded_path(rel):
                    continue
                if path.suffix.lower() not in AI_PRIORITY_B_ALLOWED_EXTS:
                    continue
                if path.stat().st_size > AI_PRIORITY_B_MAX_FILE_BYTES:
                    continue
                excluded, _ = check_ai_export_exclusion(relative_path=rel, source_path=path)
                if excluded:
                    continue
                discovered.setdefault(rel, rel)

    return sorted(discovered.keys())


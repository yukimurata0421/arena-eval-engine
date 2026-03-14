from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from arena.artifacts.models import AIManifestRecord
from arena.artifacts.policies import (
    AI_GEMINI_CORE_LIMIT,
    AI_GEMINI_PACK_KEYS,
    AI_GPT_CORE_LIMIT,
    AI_GPT_PACK_KEYS,
    AI_PACK_DETAILS_SUBDIR,
    AI_PACK_DIR_GEMINI,
    AI_PACK_DIR_GPT,
    AI_PACK_DIR_GROK,
)
from arena.artifacts.repro_stamp import resolve_generated_at
from arena.artifacts.schema import validate_pack_manifest_payload


def dedup_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique_paths: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def build_pack_source_index(
    export_dir: Path,
    records: list[AIManifestRecord],
    generated_paths: list[Path],
    deterministic: bool = False,
) -> tuple[dict[str, Path], list[Path]]:
    index: dict[str, Path] = {}
    paths: list[Path] = []

    for path in generated_paths:
        if not path.exists() or not path.is_file():
            continue
        paths.append(path)
        rel = path.relative_to(export_dir).as_posix()
        index.setdefault(rel, path)
        index.setdefault(path.name, path)

    for record in records:
        if not record.copied or not record.copied_path:
            continue
        path = export_dir / record.copied_path
        if not path.exists() or not path.is_file():
            continue
        paths.append(path)
        rel = path.relative_to(export_dir).as_posix()
        index.setdefault(rel, path)
        index.setdefault(record.relative_path, path)
        index.setdefault(Path(record.relative_path).name, path)
        index.setdefault(path.name, path)

    unique_paths = dedup_paths(paths)
    if deterministic:
        unique_paths.sort(key=lambda path: path.relative_to(export_dir).as_posix())
    return index, unique_paths


def select_pack_core_files(source_index: dict[str, Path], keys: list[str], limit: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    for key in keys:
        path = source_index.get(key)
        if path is None:
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def copy_to_dir_with_unique_name(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / src.name
    if not destination.exists():
        shutil.copy2(src, destination)
        return destination

    for index in range(1, 1000):
        candidate = dst_dir / f"{src.stem}__{index:03d}{src.suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"could not copy {src} due to destination collisions")


def copy_preserve_relative(src: Path, export_dir: Path, dst_root: Path) -> Path:
    relative = src.relative_to(export_dir)
    destination = dst_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    return destination


def write_pack_manifest(
    pack_dir: Path,
    profile_name: str,
    core_files: list[Path],
    detail_files: list[Path],
    deterministic: bool = False,
) -> None:
    manifest_path = pack_dir / "pack_manifest.txt"
    payload = {
        "profile": profile_name,
        "generated_at": resolve_generated_at(deterministic),
        "core_files": len(core_files),
        "detail_files": len(detail_files),
        "core_file_names": [path.name for path in core_files],
        "detail_file_examples": [path.relative_to(pack_dir).as_posix() for path in detail_files[:20]] or ["(none)"],
    }
    validate_pack_manifest_payload(payload)

    lines = [
        f"profile: {payload['profile']}",
        f"generated_at: {payload['generated_at']}",
        f"core_files: {payload['core_files']}",
        f"detail_files: {payload['detail_files']}",
        "core_file_names:",
    ]
    for name in payload["core_file_names"]:
        lines.append(f"- {name}")
    lines.append("detail_file_examples:")
    for relative_path in payload["detail_file_examples"]:
        lines.append(f"- {relative_path}")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")


def create_ai_review_packs(
    export_dir: Path,
    records: list[AIManifestRecord],
    generated_paths: list[Path],
    deterministic: bool = False,
) -> None:
    source_index, all_files = build_pack_source_index(
        export_dir=export_dir,
        records=records,
        generated_paths=generated_paths,
        deterministic=deterministic,
    )
    if not all_files:
        print("[WARN] ai packs skipped: no source files")
        return

    gemini_dir = export_dir / AI_PACK_DIR_GEMINI
    gemini_core_src = select_pack_core_files(source_index, AI_GEMINI_PACK_KEYS, AI_GEMINI_CORE_LIMIT)
    if gemini_dir.exists():
        shutil.rmtree(gemini_dir, ignore_errors=True)
    gemini_copied_core: list[Path] = []
    for src in gemini_core_src:
        gemini_copied_core.append(copy_to_dir_with_unique_name(src=src, dst_dir=gemini_dir))
    gemini_core_src_set = {str(path.resolve()) for path in gemini_core_src}
    gemini_detail_src = [path for path in all_files if str(path.resolve()) not in gemini_core_src_set]
    gemini_details_copied: list[Path] = []
    gemini_details_dir = gemini_dir / AI_PACK_DETAILS_SUBDIR
    for src in gemini_detail_src:
        gemini_details_copied.append(copy_preserve_relative(src=src, export_dir=export_dir, dst_root=gemini_details_dir))
    write_pack_manifest(
        pack_dir=gemini_dir,
        profile_name=AI_PACK_DIR_GEMINI,
        core_files=gemini_copied_core,
        detail_files=gemini_details_copied,
        deterministic=deterministic,
    )

    gpt_dir = export_dir / AI_PACK_DIR_GPT
    gpt_core_src = select_pack_core_files(source_index, AI_GPT_PACK_KEYS, AI_GPT_CORE_LIMIT)
    if gpt_dir.exists():
        shutil.rmtree(gpt_dir, ignore_errors=True)
    gpt_copied_core: list[Path] = []
    for src in gpt_core_src:
        gpt_copied_core.append(copy_to_dir_with_unique_name(src=src, dst_dir=gpt_dir))

    copied_record_files: list[Path] = []
    for record in records:
        if record.copied and record.copied_path:
            path = export_dir / record.copied_path
            if path.exists() and path.is_file():
                copied_record_files.append(path)
    copied_record_files = dedup_paths(copied_record_files)
    gpt_core_src_set = {str(path.resolve()) for path in gpt_core_src}
    gpt_detail_src = [path for path in copied_record_files if str(path.resolve()) not in gpt_core_src_set]
    gpt_details_copied: list[Path] = []
    gpt_details_dir = gpt_dir / AI_PACK_DETAILS_SUBDIR
    for src in gpt_detail_src:
        gpt_details_copied.append(copy_preserve_relative(src=src, export_dir=export_dir, dst_root=gpt_details_dir))
    write_pack_manifest(
        pack_dir=gpt_dir,
        profile_name=AI_PACK_DIR_GPT,
        core_files=gpt_copied_core,
        detail_files=gpt_details_copied,
        deterministic=deterministic,
    )

    grok_dir = export_dir / AI_PACK_DIR_GROK
    if grok_dir.exists():
        shutil.rmtree(grok_dir, ignore_errors=True)
    grok_copied: list[Path] = []
    for src in all_files:
        grok_copied.append(copy_preserve_relative(src=src, export_dir=export_dir, dst_root=grok_dir))
    write_pack_manifest(
        pack_dir=grok_dir,
        profile_name=AI_PACK_DIR_GROK,
        core_files=grok_copied,
        detail_files=[],
        deterministic=deterministic,
    )

    print(
        f"[OK] ai review packs: {AI_PACK_DIR_GEMINI}(core={len(gemini_copied_core)}), "
        f"{AI_PACK_DIR_GPT}(core={len(gpt_copied_core)}), "
        f"{AI_PACK_DIR_GROK}(all={len(grok_copied)})"
    )


def resolve_timestamped_export_dir(output_root: Path, prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{prefix}{ts}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base

    for index in range(1, 1000):
        candidate = output_root / f"{prefix}{ts}_{index:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    raise RuntimeError("timestamped export directory could not be resolved")

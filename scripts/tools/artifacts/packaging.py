from __future__ import annotations

import csv
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from arena.artifacts.models import AIManifestRecord
from arena.artifacts.policies import (
    AI_GEMINI_CORE_LIMIT,
    AI_GEMINI_PACK_KEYS,
    AI_GPT_CORE_LIMIT,
    AI_GPT_PACK_KEYS,
    AI_PACK_DETAILS_SUBDIR,
    AI_PACK_DIR_CLAUDE,
    AI_PACK_DIR_GEMINI,
    AI_PACK_DIR_GPT,
    AI_PACK_DIR_GROK,
    AI_PACK_MANIFESTS_DIR,
    AI_PRIORITY_DETAILS_FILL_KEYS,
    AI_PRIORITY_FILENAME_ALIASES,
)
from arena.artifacts.repro_stamp import resolve_generated_at
from arena.artifacts.schema import validate_pack_manifest_payload

AI_PACK_MANIFEST_FILENAMES = {
    AI_PACK_DIR_GEMINI: "gemini_pack.txt",
    AI_PACK_DIR_GPT: "GPT_pack.txt",
    AI_PACK_DIR_GROK: "grok_pack.txt",
    AI_PACK_DIR_CLAUDE: "claude_pack.txt",
}
AI_SELECTION_MANIFEST_FILENAMES = {
    "gemini": "gemini_selection.csv",
    "gpt": "GPT_selection.csv",
}


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_run_config(pack_dir: Path) -> dict[str, object]:
    path = pack_dir / "run_config.json"
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _config_value(config: dict[str, object], key: str) -> str:
    value = config.get(key, "unknown")
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text if text else "unknown"


def render_ai_entrypoint_markdown(
    target_ai: str,
    pack_dir: Path,
    rows: list[dict[str, object]],
) -> str:
    run_config = _load_run_config(pack_dir)
    selected_rows = sorted(
        [row for row in rows if _safe_int(row.get("copied")) == 1],
        key=lambda row: (_safe_int(row.get("rank")), str(row.get("filename", ""))),
    )
    reading_order = [str(row.get("filename", "")) for row in selected_rows if str(row.get("filename", ""))]
    included_files = sorted(
        [
            path.relative_to(pack_dir).as_posix()
            for path in pack_dir.rglob("*")
            if path.is_file()
        ]
    )

    lines = [
        "# AI Entrypoint",
        "",
        "## package purpose",
        "This artifact is an ARENA ADS-B evaluation artifact package for rapid AI-assisted review.",
        f"Target AI profile: `{target_ai}`",
        "",
        "## recommended reading order",
    ]
    if reading_order:
        for index, filename in enumerate(reading_order, start=1):
            lines.append(f"{index}. `{filename}`")
    else:
        lines.append("1. `unknown`")

    lines.extend(
        [
            "",
            "## key metrics",
            "- coverage AUC",
            "- daily quantile signature",
            "- daily coverage signature",
            "- capture rate",
            "",
            "## config resolution",
            f"- resolved_settings_path: `{_config_value(run_config, 'resolved_settings_path')}`",
            f"- resolved_phase_config_path: `{_config_value(run_config, 'resolved_phase_config_path')}`",
            f"- experimental_mode: `{_config_value(run_config, 'experimental_mode')}`",
            "",
            "## interpretation notes",
            "- traffic is a confounder; prioritize traffic-adjusted results before unadjusted comparisons.",
            "- change points indicate structural change candidates and do not prove causality by themselves.",
            "- missing summary/config files should be treated as unknown rather than hard failure.",
            "",
            "## included files",
        ]
    )
    for relative_path in included_files:
        lines.append(f"- `{relative_path}`")
    return "\n".join(lines) + "\n"


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


def build_filename_index(all_files: list[Path], export_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in all_files:
        index.setdefault(path.name, []).append(path)

    def _sort_key(path: Path) -> tuple[int, str]:
        is_root = 0 if path.parent.resolve() == export_dir.resolve() else 1
        return is_root, path.relative_to(export_dir).as_posix()

    for name in list(index.keys()):
        index[name] = sorted(dedup_paths(index[name]), key=_sort_key)
    return index


def resolve_next_source(
    filename_index: dict[str, list[Path]],
    candidate_names: list[str],
) -> tuple[Path | None, str]:
    for candidate_name in candidate_names:
        for path in filename_index.get(candidate_name, []):
            return path, candidate_name
    return None, ""


def copy_to_dir_with_name(src: Path, dst_dir: Path, filename: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / filename
    if not destination.exists():
        shutil.copy2(src, destination)
        return destination

    for index in range(1, 1000):
        candidate = dst_dir / f"{Path(filename).stem}__{index:03d}{Path(filename).suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"could not copy {src} to {filename} due to destination collisions")


def write_profile_zip(
    profile_dir: Path,
    profile_name: str,
    export_dir: Path,
    source_files: list[Path],
    deterministic: bool = False,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    zip_path = profile_dir / f"{profile_name}.zip"
    members = dedup_paths(source_files)
    if deterministic:
        members.sort(key=lambda path: path.relative_to(export_dir).as_posix())

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src in members:
            arcname = src.relative_to(export_dir).as_posix()
            if deterministic:
                zip_info = zipfile.ZipInfo(arcname)
                zip_info.date_time = (1980, 1, 1, 0, 0, 0)
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_info.external_attr = (src.stat().st_mode & 0xFFFF) << 16
                with src.open("rb") as file:
                    archive.writestr(zip_info, file.read())
            else:
                archive.write(src, arcname=arcname)
    return zip_path


def create_claude_zip_pack(export_dir: Path, source_files: list[Path], deterministic: bool = False) -> Path:
    claude_dir = export_dir / AI_PACK_DIR_CLAUDE
    if claude_dir.exists():
        shutil.rmtree(claude_dir, ignore_errors=True)
    return write_profile_zip(
        profile_dir=claude_dir,
        profile_name=AI_PACK_DIR_CLAUDE,
        export_dir=export_dir,
        source_files=source_files,
        deterministic=deterministic,
    )


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


def write_ai_selection_manifest(
    manifest_path: Path,
    target_ai: str,
    rows: list[dict[str, object]],
) -> Path:
    headers = [
        "target_ai",
        "rank",
        "filename",
        "source_subdir",
        "selected_by",
        "priority_group",
        "exists",
        "copied",
        "source_path",
        "copied_path",
        "note",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            payload = {**row}
            payload["target_ai"] = target_ai
            writer.writerow(payload)
    return manifest_path


def select_priority_pack_files(
    target_ai: str,
    pack_dir: Path,
    selection_manifest_path: Path,
    export_dir: Path,
    all_files: list[Path],
    priority_root: list[str],
    details_fill: list[str],
    filename_aliases: dict[str, list[str]],
    limit: int,
) -> tuple[list[Path], list[Path], Path]:
    pack_dir.mkdir(parents=True, exist_ok=True)
    details_dir = pack_dir / AI_PACK_DETAILS_SUBDIR
    details_dir.mkdir(parents=True, exist_ok=True)
    filename_index = build_filename_index(all_files, export_dir=export_dir)
    copied_filenames: set[str] = set()
    copied_root: list[Path] = []
    rows: list[dict[str, object]] = []
    rank = 1

    def append_row(
        filename: str,
        source_subdir: str,
        selected_by: str,
        priority_group: str,
        src: Path | None,
        resolved_name: str,
        copied_path: Path | None,
        note: str = "",
    ) -> None:
        rows.append(
            {
                "rank": rank,
                "filename": filename,
                "source_subdir": source_subdir,
                "selected_by": selected_by,
                "priority_group": priority_group,
                "exists": int(src is not None),
                "copied": int(copied_path is not None),
                "source_path": src.relative_to(export_dir).as_posix() if src else "",
                "copied_path": copied_path.relative_to(pack_dir).as_posix() if copied_path else "",
                "note": "; ".join(filter(None, [note, f"resolved_filename={resolved_name}" if resolved_name else ""])),
            }
        )

    for filename in priority_root:
        if len(copied_root) >= limit:
            break
        if filename == "ai_entrypoint.md":
            entrypoint_path = pack_dir / filename
            entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
            entrypoint_path.write_text(
                "# AI Entrypoint\n\nThis file will be regenerated by artifact packaging.\n",
                encoding="utf-8",
                newline="\n",
            )
            copied_root.append(entrypoint_path)
            copied_filenames.add(filename)
            append_row(
                filename=filename,
                source_subdir="root",
                selected_by="generated_entrypoint",
                priority_group="root_priority",
                src=entrypoint_path,
                resolved_name=filename,
                copied_path=entrypoint_path,
                note="generated_at_packaging",
            )
            rank += 1
            continue
        candidate_names = [filename, *filename_aliases.get(filename, [])]
        src, resolved_name = resolve_next_source(
            filename_index=filename_index,
            candidate_names=candidate_names,
        )
        copied_path: Path | None = None
        note = ""
        if src is not None:
            try:
                copied_path = copy_to_dir_with_name(src=src, dst_dir=pack_dir, filename=filename)
                copied_root.append(copied_path)
                copied_filenames.add(filename)
            except Exception as exc:
                note = f"copy_failed={exc}"
        append_row(
            filename=filename,
            source_subdir="root",
            selected_by="priority_root",
            priority_group="root_priority",
            src=src,
            resolved_name=resolved_name,
            copied_path=copied_path,
            note=note,
        )
        rank += 1

    if len(copied_root) < limit:
        for filename in details_fill:
            if len(copied_root) >= limit:
                break
            if filename in copied_filenames:
                continue
            candidate_names = [filename, *filename_aliases.get(filename, [])]
            src, resolved_name = resolve_next_source(
                filename_index=filename_index,
                candidate_names=candidate_names,
            )
            copied_path = None
            note = ""
            if src is not None:
                try:
                    copied_path = copy_to_dir_with_name(src=src, dst_dir=pack_dir, filename=filename)
                    copied_root.append(copied_path)
                    copied_filenames.add(filename)
                except Exception as exc:
                    note = f"copy_failed={exc}"
            append_row(
                filename=filename,
                source_subdir="details",
                selected_by="priority_details_fill",
                priority_group="details_fill",
                src=src,
                resolved_name=resolved_name,
                copied_path=copied_path,
                note=note,
            )
            rank += 1

    copied_details: list[Path] = []
    for src in all_files:
        if src.name in copied_filenames:
            continue
        copied_details.append(copy_preserve_relative(src=src, export_dir=export_dir, dst_root=details_dir))

    entrypoint_path = pack_dir / "ai_entrypoint.md"
    if entrypoint_path.exists():
        entrypoint_text = render_ai_entrypoint_markdown(
            target_ai=target_ai,
            pack_dir=pack_dir,
            rows=rows,
        )
        entrypoint_path.write_text(entrypoint_text, encoding="utf-8", newline="\n")

    manifest_path = write_ai_selection_manifest(manifest_path=selection_manifest_path, target_ai=target_ai, rows=rows)
    return copied_root, copied_details, manifest_path


def write_pack_manifest(
    manifest_path: Path,
    profile_name: str,
    pack_dir: Path,
    core_files: list[Path],
    detail_files: list[Path],
    deterministic: bool = False,
    extra_metadata: dict[str, str] | None = None,
) -> None:
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
    if extra_metadata:
        lines.append("metadata:")
        for key, value in extra_metadata.items():
            lines.append(f"- {key}: {value}")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")


def create_ai_review_packs(
    export_dir: Path,
    records: list[AIManifestRecord],
    generated_paths: list[Path],
    deterministic: bool = False,
    zip_only_profiles: bool = False,
) -> None:
    _, all_files = build_pack_source_index(
        export_dir=export_dir,
        records=records,
        generated_paths=generated_paths,
        deterministic=deterministic,
    )
    if not all_files:
        print("[WARN] ai packs skipped: no source files")
        return

    manifests_dir = export_dir / AI_PACK_MANIFESTS_DIR
    if manifests_dir.exists():
        shutil.rmtree(manifests_dir, ignore_errors=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    gemini_dir = export_dir / AI_PACK_DIR_GEMINI
    if gemini_dir.exists():
        shutil.rmtree(gemini_dir, ignore_errors=True)
    gemini_copied_core, gemini_details_copied, gemini_selection_manifest = select_priority_pack_files(
        target_ai="gemini",
        pack_dir=gemini_dir,
        selection_manifest_path=manifests_dir / AI_SELECTION_MANIFEST_FILENAMES["gemini"],
        export_dir=export_dir,
        all_files=all_files,
        priority_root=AI_GEMINI_PACK_KEYS,
        details_fill=AI_PRIORITY_DETAILS_FILL_KEYS,
        filename_aliases=AI_PRIORITY_FILENAME_ALIASES,
        limit=AI_GEMINI_CORE_LIMIT,
    )
    write_pack_manifest(
        manifest_path=manifests_dir / AI_PACK_MANIFEST_FILENAMES[AI_PACK_DIR_GEMINI],
        profile_name=AI_PACK_DIR_GEMINI,
        pack_dir=gemini_dir,
        core_files=gemini_copied_core,
        detail_files=gemini_details_copied,
        deterministic=deterministic,
        extra_metadata={
            "selection_manifest": gemini_selection_manifest.name,
            "policy": "priority_root_then_details_fill",
            "max_files": str(AI_GEMINI_CORE_LIMIT),
        },
    )

    gpt_dir = export_dir / AI_PACK_DIR_GPT
    if gpt_dir.exists():
        shutil.rmtree(gpt_dir, ignore_errors=True)
    gpt_copied_core, gpt_details_copied, gpt_selection_manifest = select_priority_pack_files(
        target_ai="gpt",
        pack_dir=gpt_dir,
        selection_manifest_path=manifests_dir / AI_SELECTION_MANIFEST_FILENAMES["gpt"],
        export_dir=export_dir,
        all_files=all_files,
        priority_root=AI_GPT_PACK_KEYS,
        details_fill=AI_PRIORITY_DETAILS_FILL_KEYS,
        filename_aliases=AI_PRIORITY_FILENAME_ALIASES,
        limit=AI_GPT_CORE_LIMIT,
    )
    write_pack_manifest(
        manifest_path=manifests_dir / AI_PACK_MANIFEST_FILENAMES[AI_PACK_DIR_GPT],
        profile_name=AI_PACK_DIR_GPT,
        pack_dir=gpt_dir,
        core_files=gpt_copied_core,
        detail_files=gpt_details_copied,
        deterministic=deterministic,
        extra_metadata={
            "selection_manifest": gpt_selection_manifest.name,
            "policy": "priority_root_then_details_fill",
            "max_files": str(AI_GPT_CORE_LIMIT),
        },
    )

    grok_dir = export_dir / AI_PACK_DIR_GROK
    if grok_dir.exists():
        shutil.rmtree(grok_dir, ignore_errors=True)
    grok_copied: list[Path] = []
    for src in all_files:
        grok_copied.append(copy_preserve_relative(src=src, export_dir=export_dir, dst_root=grok_dir))
    write_pack_manifest(
        manifest_path=manifests_dir / AI_PACK_MANIFEST_FILENAMES[AI_PACK_DIR_GROK],
        profile_name=AI_PACK_DIR_GROK,
        pack_dir=grok_dir,
        core_files=grok_copied,
        detail_files=[],
        deterministic=deterministic,
        extra_metadata={"policy": "all_files_uncompressed"},
    )
    claude_zip = create_claude_zip_pack(export_dir=export_dir, source_files=all_files, deterministic=deterministic)
    write_pack_manifest(
        manifest_path=manifests_dir / AI_PACK_MANIFEST_FILENAMES[AI_PACK_DIR_CLAUDE],
        profile_name=AI_PACK_DIR_CLAUDE,
        pack_dir=export_dir / AI_PACK_DIR_CLAUDE,
        core_files=[claude_zip],
        detail_files=[],
        deterministic=deterministic,
        extra_metadata={
            "zip_name": claude_zip.name,
            "policy": "folder_zip",
        },
    )

    print(
        f"[OK] ai review packs: {AI_PACK_DIR_GEMINI}(selected={len(gemini_copied_core)}, details={len(gemini_details_copied)}), "
        f"{AI_PACK_DIR_GPT}(selected={len(gpt_copied_core)}, details={len(gpt_details_copied)}), "
        f"{AI_PACK_DIR_GROK}(all={len(grok_copied)}), "
        f"{AI_PACK_DIR_CLAUDE}(zip={claude_zip.name})"
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

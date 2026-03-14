from __future__ import annotations

from pathlib import Path
from typing import Sequence

from arena.pipeline.stages import Step, _resolve_expected_path, validate_outputs


def latest_dependency_mtime(output_root_native: Path, dependencies: Sequence[str]) -> float | None:
    if not dependencies:
        return None

    mtimes: list[float] = []
    for dep in dependencies:
        fp = _resolve_expected_path(output_root_native, dep)
        if not fp.exists():
            return None
        try:
            if fp.is_dir():
                files = [p for p in fp.rglob("*") if p.is_file()]
                if not files:
                    return None
                mtimes.append(max(p.stat().st_mtime for p in files))
            else:
                mtimes.append(fp.stat().st_mtime)
        except Exception:
            return None
    return max(mtimes) if mtimes else None


def should_skip_existing(
    skip_existing: bool,
    step: Step,
    output_root_native: Path,
    data_root_native: Path | None = None,
) -> bool:
    if not skip_existing or not step.expected_outputs:
        return False
    if step.always_run_when_skip_existing:
        # Explicit per-step policy for live/rolling inputs.
        return False

    dep_min_mtime = latest_dependency_mtime(output_root_native, step.depends_on_outputs)
    if step.depends_on_outputs and dep_min_mtime is None:
        return False

    ok, _missing = validate_outputs(
        output_root_native,
        step.expected_outputs,
        step.expected_min_bytes,
        min_mtime=dep_min_mtime,
        data_root_native=data_root_native,
    )
    return ok


def can_soft_fail(
    step: Step,
    output_root_native: Path,
    min_mtime: float | None,
    data_root_native: Path | None = None,
) -> tuple[bool, list[str]]:
    if not step.soft_fail_on_error:
        return False, []
    if not step.expected_outputs:
        return True, []

    check_mtime = None if step.soft_fail_accept_stale_outputs else min_mtime
    ok, missing = validate_outputs(
        output_root_native,
        step.expected_outputs,
        step.expected_min_bytes,
        min_mtime=check_mtime,
        data_root_native=data_root_native,
    )
    return ok, missing


def resolve_input_probe(step: Step, data_root_native: Path) -> tuple[Path, str]:
    raw_input_dir = step.input_dir.strip()
    input_dir = Path(raw_input_dir) if raw_input_dir else data_root_native
    if not input_dir.is_absolute():
        input_dir = data_root_native / input_dir
    pattern = step.input_pattern.strip() or "*"

    if step.extra_args:
        for i, arg in enumerate(step.extra_args):
            if step.input_dir_arg and arg == step.input_dir_arg and i + 1 < len(step.extra_args):
                input_dir = Path(step.extra_args[i + 1])
            elif step.input_pattern_arg and arg == step.input_pattern_arg and i + 1 < len(step.extra_args):
                pattern = step.extra_args[i + 1]
    return input_dir, pattern


def should_skip_no_inputs(step: Step, data_root_native: Path) -> tuple[bool, Path, str]:
    if not step.skip_if_no_inputs:
        return False, Path(), ""

    input_dir, pattern = resolve_input_probe(step, data_root_native)
    try:
        has_inputs = any(input_dir.glob(pattern))
    except Exception:
        has_inputs = False
    return (not has_inputs), input_dir, pattern

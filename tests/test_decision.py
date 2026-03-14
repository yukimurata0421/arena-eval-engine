from __future__ import annotations

import time
from pathlib import Path

from arena.pipeline.decision import (
    can_soft_fail,
    latest_dependency_mtime,
    resolve_input_probe,
    should_skip_existing,
    should_skip_no_inputs,
)
from arena.pipeline.stages import Step


def test_latest_dependency_mtime_returns_none_on_missing(tmp_path: Path) -> None:
    assert latest_dependency_mtime(tmp_path, ["missing.csv"]) is None


def test_latest_dependency_mtime_reads_files_and_dirs(tmp_path: Path) -> None:
    f = tmp_path / "dep.csv"
    f.write_text("x", encoding="utf-8")
    d = tmp_path / "dep_dir"
    d.mkdir()
    (d / "child.txt").write_text("x", encoding="utf-8")
    mtime = latest_dependency_mtime(tmp_path, ["dep.csv", "dep_dir"])
    assert isinstance(mtime, float)


def test_should_skip_existing_true_when_outputs_present(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    out.write_text("x" * 100, encoding="utf-8")
    step = Step(stage=1, script_rel="a.py", label="x", expected_outputs=["out.csv"], expected_min_bytes=50)
    assert should_skip_existing(True, step, tmp_path) is True


def test_should_skip_existing_false_for_always_run(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    out.write_text("x" * 100, encoding="utf-8")
    step = Step(
        stage=1,
        script_rel="a.py",
        label="x",
        expected_outputs=["out.csv"],
        expected_min_bytes=50,
        always_run_when_skip_existing=True,
    )
    assert should_skip_existing(True, step, tmp_path) is False


def test_can_soft_fail_accepts_no_expected_outputs(tmp_path: Path) -> None:
    step = Step(stage=1, script_rel="a.py", label="x", soft_fail_on_error=True)
    ok, missing = can_soft_fail(step, tmp_path, min_mtime=time.time())
    assert ok is True
    assert missing == []


def test_resolve_input_probe_uses_relative_input_dir_and_defaults(tmp_path: Path) -> None:
    step = Step(
        stage=1,
        script_rel="a.py",
        label="x",
        skip_if_no_inputs=True,
        input_dir="plao_pos",
        input_pattern="pos_*.jsonl",
    )
    input_dir, pattern = resolve_input_probe(step, tmp_path)
    assert input_dir == (tmp_path / "plao_pos")
    assert pattern == "pos_*.jsonl"


def test_resolve_input_probe_allows_extra_args_override(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom"
    step = Step(
        stage=1,
        script_rel="a.py",
        label="x",
        skip_if_no_inputs=True,
        input_dir="plao_pos",
        input_pattern="pos_*.jsonl",
        extra_args=["--input-dir", str(custom_dir), "--pattern", "x_*.jsonl"],
    )
    input_dir, pattern = resolve_input_probe(step, tmp_path)
    assert input_dir == custom_dir
    assert pattern == "x_*.jsonl"


def test_should_skip_no_inputs_true_when_probe_empty(tmp_path: Path) -> None:
    step = Step(
        stage=1,
        script_rel="a.py",
        label="x",
        skip_if_no_inputs=True,
        input_dir="plao_pos",
        input_pattern="pos_*.jsonl",
    )
    skip, input_dir, _pattern = should_skip_no_inputs(step, tmp_path)
    assert skip is True
    assert input_dir == (tmp_path / "plao_pos")

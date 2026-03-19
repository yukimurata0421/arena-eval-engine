from __future__ import annotations

from pathlib import Path

from arena.pipeline.backend import Backend
from arena.pipeline.runner import PipelineRunner
from arena.pipeline.stages import RunRecord, Step


def _make_record(*, status: str, stage: int = 1, label: str = "x", step_code: str = "S1-01", missing: list[str] | None = None) -> RunRecord:
    rc = 0 if status == "OK" else 1
    return RunRecord(
        ts_start="2026-01-01T00:00:00",
        ts_end="2026-01-01T00:00:01",
        backend="native",
        stage=stage,
        label=label,
        script_rel="x.py",
        status=status,
        elapsed_s=1.0,
        returncode=rc,
        cmd=["python", "x.py"],
        expected_outputs=[],
        outputs_ok=(status == "OK"),
        missing_outputs=missing or [],
        step_code=step_code,
        error_code="",
        stderr_tail="",
        stdout_tail="",
    )


def test_print_summary_aggregates_status_counts_and_codes(tmp_path: Path, caplog) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    backend = Backend(kind="native", scripts_root_native=scripts_root, output_root_native=output_root, data_root_native=data_root)
    runner = PipelineRunner(
        backend=backend,
        dry_run=True,
        validate=False,
        jsonl_log_path=output_root / "performance" / "pipeline_runs.jsonl",
        jax_platforms="cpu",
        skip_existing=False,
        fail_fast=False,
        phase_config_path="",
        workers=1,
        steps=[
            Step(stage=1, script_rel="x.py", label="x", error_code_base="S1-01"),
            Step(stage=1, script_rel="y.py", label="y", error_code_base="S1-02"),
        ],
    )

    # Patch code mapping to focus on aggregation logic.
    runner._error_code_for_record = lambda rec: f"{rec.step_code}-E10"  # type: ignore[method-assign]

    runner.records = [
        _make_record(status="OK", step_code="S1-01", label="ok"),
        _make_record(status="WARN", step_code="S1-02", label="warn"),
        _make_record(status="FAIL_OUTPUT", step_code="S1-02", label="out", missing=["missing.txt"]),
    ]

    with caplog.at_level("INFO"):
        runner.print_summary()

    text = "\n".join([r.getMessage() for r in caplog.records])
    assert "pipeline aggregation" in text
    assert "Result:" in text
    assert "1 OK" in text
    assert "1 WARN" in text
    assert "1 NG(output)" in text
    # Ensure error-code section appears for non-OK records
    assert "Error code details" in text
    assert "S1-02-E10" in text


def test_write_error_code_report_includes_catalog_and_detected_issues(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts"
    output_root = tmp_path / "output"
    data_root = tmp_path / "data"
    scripts_root.mkdir()
    output_root.mkdir()
    data_root.mkdir()

    backend = Backend(kind="native", scripts_root_native=scripts_root, output_root_native=output_root, data_root_native=data_root)
    runner = PipelineRunner(
        backend=backend,
        dry_run=True,
        validate=False,
        jsonl_log_path=output_root / "performance" / "pipeline_runs.jsonl",
        jax_platforms="cpu",
        skip_existing=False,
        fail_fast=False,
        phase_config_path="",
        workers=1,
        steps=[
            Step(stage=1, script_rel="x.py", label="x", error_code_base="S1-01"),
        ],
    )
    runner._error_code_for_record = lambda rec: f"{rec.step_code}-E31"  # type: ignore[method-assign]
    runner.records = [
        _make_record(status="TIMEOUT", step_code="S1-01", label="timeout"),
    ]

    report_path = runner.write_error_code_report()
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Step Code Catalog" in text
    assert "S1-01 | S1 | x" in text
    assert "Detected Issues" in text
    assert "S1-01-E31" in text


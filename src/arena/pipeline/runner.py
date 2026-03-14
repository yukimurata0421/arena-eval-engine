from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

from arena.pipeline.backend import BATCH_ENV, Backend, now_iso, resolve_default_workers, tail_text
from arena.pipeline.decision import can_soft_fail, should_skip_existing, should_skip_no_inputs
from arena.pipeline.error_policy import error_code_for_record, recommended_actions, summarize_issue_reason
from arena.pipeline.record_io import append_jsonl, build_config_snapshot_payload, build_run_record_payload
from arena.pipeline.stages import STAGE_NAMES, RunRecord, Step, validate_outputs

class PipelineRunner:
    def __init__(
        self,
        backend: Backend,
        dry_run: bool,
        validate: bool,
        jsonl_log_path: Path,
        jax_platforms: str,
        skip_existing: bool,
        fail_fast: bool,
        phase_config_path: str = "",
        workers: int = 0,
        steps: Sequence[Step] | None = None,
    ) -> None:
        self._phase_config_path = phase_config_path
        self.backend = backend
        self.dry_run = dry_run
        self.validate = validate
        self.jsonl_log_path = jsonl_log_path
        self.skip_existing = skip_existing
        self.fail_fast = fail_fast
        self.workers = workers if workers > 0 else resolve_default_workers()

        src_root = self.backend.scripts_root_native.parent / "src"
        if src_root.exists():
            existing = os.environ.get("PYTHONPATH", "")
            py_path = f"{src_root}{os.pathsep}{existing}" if existing else str(src_root)
        else:
            py_path = os.environ.get("PYTHONPATH", "")
        self.env = {
            **os.environ,
            **BATCH_ENV,
            "JAX_PLATFORMS": jax_platforms,
            "ADSB_PHASE_CONFIG": self._phase_config_path,
            "ARENA_MAX_WORKERS": str(self.workers),
            "ADSB_MAX_WORKERS": str(self.workers),
            "ADSB_PHASE_CHAINS": str(self.workers),
            "PYTHONPATH": py_path,
        }
        self.records: list[RunRecord] = []
        self.step_catalog: dict[str, tuple[int, str]] = {}
        if steps:
            for st in steps:
                code = st.error_code_base or ""
                if code:
                    self.step_catalog[code] = (st.stage, st.label)

        # ensure log dir exists
        self.jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, rec: RunRecord) -> None:
        if not rec.error_code:
            rec.error_code = self._error_code_for_record(rec)
        payload = build_run_record_payload(rec)
        append_jsonl(self.jsonl_log_path, payload)

    def log_config_snapshot(self, snapshot: dict) -> None:
        payload = build_config_snapshot_payload(
            ts=now_iso(),
            backend_desc=self.backend.describe(),
            snapshot=snapshot,
        )
        append_jsonl(self.jsonl_log_path, payload)

    def _should_skip_existing(self, step: Step) -> bool:
        return should_skip_existing(
            self.skip_existing,
            step,
            self.backend.output_root_native,
            data_root_native=self.backend.data_root_native,
        )

    def _attach_codes(self, rec: RunRecord, step: Step) -> None:
        rec.step_code = step.error_code_base or f"S{step.stage}-00"
        rec.error_code = self._error_code_for_record(rec)

    def _error_code_for_record(self, rec: RunRecord) -> str:
        return error_code_for_record(rec)

    def _can_soft_fail(self, step: Step, min_mtime: float | None) -> tuple[bool, list[str]]:
        return can_soft_fail(
            step,
            self.backend.output_root_native,
            min_mtime,
            data_root_native=self.backend.data_root_native,
        )

    def _should_skip_no_inputs(self, step: Step) -> tuple[bool, Path]:
        skip, input_dir, _pattern = should_skip_no_inputs(step, self.backend.data_root_native)
        return skip, input_dir

    def run_step(self, step: Step) -> bool:
        # normalize script path to posix
        script_rel = step.script_rel.replace("\\", "/")

        # skip existing outputs (optional)
        if self._should_skip_existing(step):
            rec = RunRecord(
                ts_start=now_iso(),
                ts_end=now_iso(),
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="SKIP(existing)",
                elapsed_s=0.0,
                returncode=None,
                cmd=[],
                expected_outputs=list(step.expected_outputs),
                outputs_ok=True,
                missing_outputs=[],
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)
            print(f"    - [SKIP] {step.label} (outputs already exist)")
            return True

        cmd, cwd = self.backend.build_script_cmd(script_rel, step.extra_args or None)

        # dry-run
        if self.dry_run:
            rec = RunRecord(
                ts_start=now_iso(),
                ts_end=now_iso(),
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="DRY",
                elapsed_s=0.0,
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=True,
                missing_outputs=[],
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)
            est = f" (~{step.est_s}s)" if step.est_s else ""
            print(f"    [DRY] {step.label}{est}: {script_rel}")
            return True

        # optional skip when required inputs are absent
        skip_no_inputs, input_dir = self._should_skip_no_inputs(step)
        if skip_no_inputs:
            rec = RunRecord(
                ts_start=now_iso(),
                ts_end=now_iso(),
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status="SKIP(no input)",
                elapsed_s=0.0,
                returncode=None,
                cmd=[],
                expected_outputs=list(step.expected_outputs),
                outputs_ok=True,
                missing_outputs=[],
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)
            print(f"    - [SKIP] {step.label} (no input files: {input_dir})")
            return True

        # existence check for native backend
        if self.backend.kind == "native":
            sp = self.backend.scripts_root_native / script_rel
            if not sp.exists():
                rec = RunRecord(
                    ts_start=now_iso(),
                    ts_end=now_iso(),
                    backend=self.backend.describe(),
                    stage=step.stage,
                    label=step.label,
                    script_rel=script_rel,
                    status="NOT_FOUND",
                    elapsed_s=0.0,
                    returncode=None,
                    cmd=cmd,
                    expected_outputs=list(step.expected_outputs),
                    outputs_ok=False,
                    missing_outputs=[str(sp)],
                    stderr_tail="",
                    stdout_tail="",
                )
                self._attach_codes(rec, step)
                self.records.append(rec)
                self._append_jsonl(rec)
                print(f"    ? [NOT_FOUND] {step.label}: {script_rel}")
                return (not step.critical) and (not self.fail_fast)

        # run
        t0 = time.time()
        ts0 = now_iso()
        est = f" (~{step.est_s}s)" if step.est_s else ""
        print(f"    > {step.label}{est} ...", end=" ", flush=True)

        try:
            env = {**self.env, **(step.env_overrides or {})}
            if self.backend.kind == "wsl" and step.env_overrides:
                exports = "; ".join(
                    [f"export {k}={shlex.quote(v)}" for k, v in step.env_overrides.items()]
                )
                cmd = list(cmd)
                cmd[-1] = f"{exports}; {cmd[-1]}"
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                input=step.input_text,
                capture_output=True,
                text=True,
                timeout=step.timeout_s,
                encoding="utf-8",
                errors="replace",
            )
            elapsed = time.time() - t0
            ts1 = now_iso()

            status = "OK" if proc.returncode == 0 else "FAIL"
            outputs_ok = True
            missing = []

            if self.validate and step.expected_outputs:
                outputs_ok, missing = validate_outputs(
                    self.backend.output_root_native,
                    step.expected_outputs,
                    step.expected_min_bytes,
                    min_mtime=t0 - 1.0,
                    data_root_native=self.backend.data_root_native,
                )
                if status == "OK" and not outputs_ok:
                    status = "FAIL_OUTPUT"
            if status in ("FAIL", "FAIL_OUTPUT"):
                soft_ok, _soft_missing = self._can_soft_fail(step, t0 - 1.0)
                if soft_ok:
                    status = "WARN"
                    outputs_ok = True
                    missing = []

            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status=status,
                elapsed_s=round(elapsed, 2),
                returncode=proc.returncode,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=outputs_ok,
                missing_outputs=missing,
                stderr_tail=tail_text(proc.stderr or "", 1400),
                stdout_tail=tail_text(proc.stdout or "", 1400),
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)

            if status == "OK":
                print(f"OK（{elapsed:.1f}s）")
                return True
            if status == "WARN":
                print(f"WARN（{elapsed:.1f}s）")
                return True

            # failure printing (tail)
            print(f"NG（{elapsed:.1f}s）")
            if status == "FAIL_OUTPUT" and missing:
                print("      [output validation failed]")
                for m in missing[:3]:
                    print(f"      - {m}")
                if len(missing) > 3:
                    print(f"      ... (+{len(missing)-3} more)")
            if proc.stderr:
                lines = proc.stderr.strip().splitlines()
                for line in lines[-3:]:
                    print(f"      {line}")

            # control flow
            if step.critical or self.fail_fast:
                return False
            return True

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            ts1 = now_iso()
            soft_ok, _soft_missing = self._can_soft_fail(step, t0 - 1.0)
            status = "WARN" if soft_ok else "TIMEOUT"
            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status=status,
                elapsed_s=round(elapsed, 2),
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=soft_ok,
                missing_outputs=[],
                stderr_tail="",
                stdout_tail="",
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)
            if status == "WARN":
                print(f"WARN（{step.timeout_s}s timeout）")
                return True
            print(f"NG（{step.timeout_s}s）")
            return (not step.critical) and (not self.fail_fast)

        except Exception as e:
            elapsed = time.time() - t0
            ts1 = now_iso()
            soft_ok, _soft_missing = self._can_soft_fail(step, t0 - 1.0)
            status = "WARN" if soft_ok else "ERROR"
            rec = RunRecord(
                ts_start=ts0,
                ts_end=ts1,
                backend=self.backend.describe(),
                stage=step.stage,
                label=step.label,
                script_rel=script_rel,
                status=status,
                elapsed_s=round(elapsed, 2),
                returncode=None,
                cmd=cmd,
                expected_outputs=list(step.expected_outputs),
                outputs_ok=soft_ok,
                missing_outputs=[],
                stderr_tail=str(e),
                stdout_tail="",
            )
            self._attach_codes(rec, step)
            self.records.append(rec)
            self._append_jsonl(rec)
            if status == "WARN":
                print(f"WARN ({e})")
                return True
            print(f"ERROR ({e})")
            return (not step.critical) and (not self.fail_fast)

    def print_summary(self) -> None:
        print("\n" + "=" * 78)
        print("Pipeline Summary")
        print("=" * 78)

        total = sum(r.elapsed_s for r in self.records)
        counts: dict[str, int] = {}
        for r in self.records:
            k = r.status.split("(")[0]  # SKIP(x) -> SKIP
            counts[k] = counts.get(k, 0) + 1

        for r in self.records:
            icon = {
                "OK": "OK",
                "WARN": "WARN",
                "FAIL": "NG",
                "FAIL_OUTPUT": "NG*",
                "TIMEOUT": "TO",
                "ERROR": "!!",
                "NOT_FOUND": "?",
                "DRY": "DRY",
            }.get(r.status.split("(")[0], "-")
            t = f"{r.elapsed_s:7.1f}s" if r.elapsed_s > 0 else "        "
            stage_name = STAGE_NAMES.get(r.stage, f"Stage {r.stage}")
            code = self._error_code_for_record(r)
            status_disp = f"{r.status}/{code}" if code else r.status
            print(f"  {icon} {t}  [{status_disp:<18}] S{r.stage}({stage_name}) {r.label}")

        issues = [r for r in self.records if r.status.split("(")[0] in ("WARN", "FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
        if issues:
            print("-" * 78)
            print("  Error Code Details")
            for r in issues:
                stage_name = STAGE_NAMES.get(r.stage, f"Stage {r.stage}")
                reason = self._summarize_warn_reason(r)
                code = self._error_code_for_record(r) or "N/A"
                print(f"  - {code} | S{r.stage}({stage_name}) {r.label}: {reason}")

            actions = self._recommended_actions(issues)
            if actions:
                print("-" * 78)
                print("  Recommended Actions")
                for a in actions:
                    print(f"  - {a}")

        print("-" * 78)
        parts = []
        for key in ["OK", "WARN", "FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "SKIP", "DRY", "NOT_FOUND"]:
            if key in counts:
                parts.append(f"{counts[key]} {key}")
        print("  " + " / ".join(parts))
        print(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
        print(f"  Log: {self.jsonl_log_path}")
        print("=" * 78)

    def write_error_code_report(self) -> Path:
        report_path = self.backend.output_root_native / "performance" / "pipeline_error_codes_latest.txt"
        issues = [r for r in self.records if r.status.split("(")[0] in ("WARN", "FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
        lines = [
            "=" * 100,
            "Pipeline Error Code Report",
            "=" * 100,
            f"generated_at: {now_iso()}",
            f"log_jsonl: {self.jsonl_log_path}",
            "",
            "Code format: <STEP_BASE>-<TYPE><NN>",
            "  TYPE=W: warning, E: error",
            "  Common suffixes: E10 generic fail, E20 output invalid, E21 stale, E22 too small, E23 empty dir, E31 timeout, E32 not found, E33 exception",
            "",
            "Step Code Catalog",
            "-" * 100,
        ]
        if self.step_catalog:
            for code in sorted(self.step_catalog.keys()):
                stage, label = self.step_catalog[code]
                lines.append(f"{code} | S{stage} | {label}")
        else:
            lines.append("N/A")
        lines.extend(["", "Detected Issues"])
        if not issues:
            lines.append("NO_ISSUES")
        else:
            lines.append("code | status | stage | label | reason")
            lines.append("-" * 100)
            for r in issues:
                code = self._error_code_for_record(r) or "N/A"
                reason = self._summarize_warn_reason(r)
                lines.append(f"{code} | {r.status} | S{r.stage} | {r.label} | {reason}")
            actions = self._recommended_actions(issues)
            if actions:
                lines.extend(["", "Recommended Actions", "-" * 100])
                for a in actions:
                    lines.append(f"- {a}")
        lines.append("")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    @staticmethod
    def _summarize_warn_reason(rec: RunRecord) -> str:
        return summarize_issue_reason(rec)

    def _recommended_actions(self, issues: Sequence[RunRecord]) -> list[str]:
        return recommended_actions(issues)

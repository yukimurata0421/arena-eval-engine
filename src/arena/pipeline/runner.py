from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from arena.log import get_logger
from arena.pipeline.backend import BATCH_ENV, Backend, now_iso, resolve_default_workers, tail_text
from arena.pipeline.decision import can_soft_fail, should_skip_existing, should_skip_no_inputs
from arena.pipeline.error_policy import error_code_for_record, recommended_actions, summarize_issue_reason
from arena.pipeline.record_io import append_jsonl, build_config_snapshot_payload, build_run_record_payload
from arena.pipeline.stages import STAGE_NAMES, RunRecord, Step, validate_outputs

logger = get_logger(__name__)

TAIL_MAX_CHARS = 1400
TRACEBACK_MAX_CHARS = 500


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
        settings_path: str = "",
        workers: int = 0,
        steps: Sequence[Step] | None = None,
    ) -> None:
        self._phase_config_path = phase_config_path
        self._settings_path = settings_path
        self.backend = backend
        self.dry_run = dry_run
        self.validate = validate
        self.jsonl_log_path = jsonl_log_path
        self.skip_existing = skip_existing
        self.fail_fast = fail_fast
        self.workers = workers if workers > 0 else resolve_default_workers()
        self._record_lock = threading.Lock()
        self._print_lock = threading.Lock()

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
            "ARENA_PHASE_CONFIG": self._phase_config_path,
            "ARENA_SETTINGS": self._settings_path,
            "ADSB_SETTINGS": self._settings_path,
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

        self.jsonl_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # RunRecord factory
    # ------------------------------------------------------------------
    @staticmethod
    def _make_record(
        step: Step,
        script_rel: str,
        backend_desc: str,
        *,
        status: str,
        ts_start: str = "",
        ts_end: str = "",
        elapsed_s: float = 0.0,
        returncode: int | None = None,
        cmd: list[str] | None = None,
        outputs_ok: bool = True,
        missing_outputs: list[str] | None = None,
        stderr_tail: str = "",
        stdout_tail: str = "",
    ) -> RunRecord:
        now = ts_start or now_iso()
        return RunRecord(
            ts_start=now,
            ts_end=ts_end or now,
            backend=backend_desc,
            stage=step.stage,
            label=step.label,
            script_rel=script_rel,
            status=status,
            elapsed_s=elapsed_s,
            returncode=returncode,
            cmd=cmd or [],
            expected_outputs=list(step.expected_outputs),
            outputs_ok=outputs_ok,
            missing_outputs=missing_outputs or [],
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_jsonl(self, rec: RunRecord) -> None:
        if not rec.error_code:
            rec.error_code = self._error_code_for_record(rec)
        payload = build_run_record_payload(rec)
        with self._record_lock:
            self.records.append(rec)
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

    def _record_and_log(self, step: Step, rec: RunRecord, msg: str) -> None:
        self._attach_codes(rec, step)
        self._append_jsonl(rec)
        with self._print_lock:
            logger.info(msg)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------
    def run_step(self, step: Step) -> bool:
        script_rel = step.script_rel.replace("\\", "/")
        backend_desc = self.backend.describe()

        if self._should_skip_existing(step):
            rec = self._make_record(step, script_rel, backend_desc, status="SKIP(existing)")
            self._record_and_log(step, rec, f"    [SKIP] 出力既存 - {step.label}")
            return True

        cmd, cwd = self.backend.build_script_cmd(script_rel, step.extra_args or None)

        if self.dry_run:
            rec = self._make_record(step, script_rel, backend_desc, status="DRY", cmd=cmd)
            est_str = f" (予測 約{step.est_s}s)" if step.est_s else ""
            self._record_and_log(step, rec, f"    [DRY] {step.label}{est_str}: {script_rel}")
            return True

        skip_no_inputs, _input_dir = self._should_skip_no_inputs(step)
        if skip_no_inputs:
            rec = self._make_record(step, script_rel, backend_desc, status="SKIP(no input)")
            self._record_and_log(step, rec, f"    [SKIP] 入力なし - {step.label}")
            return True

        if self.backend.kind == "native":
            sp = self.backend.scripts_root_native / script_rel
            if not sp.exists():
                rec = self._make_record(
                    step,
                    script_rel,
                    backend_desc,
                    status="NOT_FOUND",
                    cmd=cmd,
                    outputs_ok=False,
                    missing_outputs=[str(sp)],
                )
                self._record_and_log(step, rec, f"    [NG] NOT_FOUND - {step.label}")
                return (not step.critical) and (not self.fail_fast)

        return self._execute_step(step, script_rel, backend_desc, cmd, cwd)

    def _execute_step(
        self,
        step: Step,
        script_rel: str,
        backend_desc: str,
        cmd: list[str],
        cwd: str | None,
    ) -> bool:
        t0 = time.time()
        ts0 = now_iso()
        est_str = f" (予測 約{step.est_s}s)" if step.est_s else ""
        with self._print_lock:
            logger.info("    > %s%s ...", step.label, est_str)

        try:
            env = {**self.env, **(step.env_overrides or {})}
            if self.backend.kind == "wsl" and step.env_overrides:
                exports = "; ".join([f"export {k}={shlex.quote(v)}" for k, v in step.env_overrides.items()])
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
            missing: list[str] = []

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

            rec = self._make_record(
                step,
                script_rel,
                backend_desc,
                status=status,
                ts_start=ts0,
                ts_end=ts1,
                elapsed_s=round(elapsed, 2),
                returncode=proc.returncode,
                cmd=cmd,
                outputs_ok=outputs_ok,
                missing_outputs=missing,
                stderr_tail=tail_text(proc.stderr or "", TAIL_MAX_CHARS),
                stdout_tail=tail_text(proc.stdout or "", TAIL_MAX_CHARS),
            )
            self._attach_codes(rec, step)
            self._append_jsonl(rec)

            return self._report_execution_result(step, status, elapsed, missing, proc.stderr)

        except subprocess.TimeoutExpired:
            return self._handle_timeout(step, script_rel, backend_desc, cmd, t0, ts0)

        except OSError as e:
            return self._handle_exception(step, script_rel, backend_desc, cmd, t0, ts0, e)

    def _report_execution_result(
        self,
        step: Step,
        status: str,
        elapsed: float,
        missing: list[str],
        stderr: str | None,
    ) -> bool:
        if status == "OK":
            with self._print_lock:
                logger.info("    [OK] 実績 %.1fs - %s", elapsed, step.label)
            return True
        if status == "WARN":
            with self._print_lock:
                logger.warning("    [WARN] 実績 %.1fs - %s", elapsed, step.label)
            return True

        with self._print_lock:
            logger.error("    [NG] 実績 %.1fs - %s", elapsed, step.label)
            if status == "FAIL_OUTPUT" and missing:
                logger.error("      [出力検証に失敗]")
                for m in missing[:3]:
                    logger.error("      - %s", m)
                if len(missing) > 3:
                    logger.error("      ... (+%d more)", len(missing) - 3)
            if stderr:
                for line in stderr.strip().splitlines()[-3:]:
                    logger.error("      %s", line)

        return not (step.critical or self.fail_fast)

    def _handle_timeout(
        self,
        step: Step,
        script_rel: str,
        backend_desc: str,
        cmd: list[str],
        t0: float,
        ts0: str,
    ) -> bool:
        elapsed = time.time() - t0
        ts1 = now_iso()
        soft_ok, _soft_missing = self._can_soft_fail(step, t0 - 1.0)
        status = "WARN" if soft_ok else "TIMEOUT"
        rec = self._make_record(
            step,
            script_rel,
            backend_desc,
            status=status,
            ts_start=ts0,
            ts_end=ts1,
            elapsed_s=round(elapsed, 2),
            cmd=cmd,
            outputs_ok=soft_ok,
        )
        self._attach_codes(rec, step)
        self._append_jsonl(rec)
        with self._print_lock:
            if status == "WARN":
                logger.warning("    [WARN] 実績 %ss (timeout) - %s", step.timeout_s, step.label)
                return True
            logger.error("    [NG] 実績 %ss (timeout) - %s", step.timeout_s, step.label)
        return (not step.critical) and (not self.fail_fast)

    def _handle_exception(
        self,
        step: Step,
        script_rel: str,
        backend_desc: str,
        cmd: list[str],
        t0: float,
        ts0: str,
        exc: Exception,
    ) -> bool:
        elapsed = time.time() - t0
        ts1 = now_iso()
        soft_ok, _soft_missing = self._can_soft_fail(step, t0 - 1.0)
        status = "WARN" if soft_ok else "ERROR"
        full_tb = traceback.format_exc()
        rec = self._make_record(
            step,
            script_rel,
            backend_desc,
            status=status,
            ts_start=ts0,
            ts_end=ts1,
            elapsed_s=round(elapsed, 2),
            cmd=cmd,
            outputs_ok=soft_ok,
            stderr_tail=full_tb[:TRACEBACK_MAX_CHARS] if len(full_tb) > TRACEBACK_MAX_CHARS else full_tb,
        )
        self._attach_codes(rec, step)
        self._append_jsonl(rec)
        with self._print_lock:
            if status == "WARN":
                logger.warning("    [WARN] %s - %s", exc, step.label)
                return True
            logger.error("    [NG] エラー %s - %s", exc, step.label)
        return (not step.critical) and (not self.fail_fast)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        logger.info("\n" + "=" * 78)
        logger.info("パイプライン集計")
        logger.info("=" * 78)

        total = sum(r.elapsed_s for r in self.records)
        counts: dict[str, int] = {}
        for r in self.records:
            k = r.status.split("(")[0]
            counts[k] = counts.get(k, 0) + 1

        for r in self.records:
            raw = r.status.split("(")[0]
            tag = {
                "OK": "[OK]   ",
                "WARN": "[WARN] ",
                "FAIL": "[NG]   ",
                "FAIL_OUTPUT": "[NG*]  ",
                "TIMEOUT": "[NG]   ",
                "ERROR": "[NG]   ",
                "NOT_FOUND": "[NG]   ",
                "SKIP": "[SKIP] ",
                "DRY": "[DRY]  ",
            }.get(raw, "[?]    ")
            t = f"{r.elapsed_s:7.1f}s" if r.elapsed_s > 0 else "        "
            stage_name = STAGE_NAMES.get(r.stage, f"Stage {r.stage}")
            code = self._error_code_for_record(r)
            code_str = f" {code}" if code else ""
            logger.info("  %s %s  S%d(%s) %s%s", tag, t, r.stage, stage_name, r.label, code_str)

        issues = [r for r in self.records if r.status.split("(")[0] in ("WARN", "FAIL", "FAIL_OUTPUT", "TIMEOUT", "ERROR", "NOT_FOUND")]
        if issues:
            logger.info("-" * 78)
            logger.info("  エラーコード詳細")
            for r in issues:
                stage_name = STAGE_NAMES.get(r.stage, f"Stage {r.stage}")
                reason = self._summarize_warn_reason(r)
                code = self._error_code_for_record(r) or "N/A"
                logger.info("  - %s | S%d(%s) %s: %s", code, r.stage, stage_name, r.label, reason)

            actions = self._recommended_actions(issues)
            if actions:
                logger.info("-" * 78)
                logger.info("  推奨アクション")
                for a in actions:
                    logger.info("  - %s", a)

        logger.info("-" * 78)
        parts = []
        for key, label in [
            ("OK", "OK"),
            ("WARN", "WARN"),
            ("FAIL", "NG"),
            ("FAIL_OUTPUT", "NG(出力)"),
            ("TIMEOUT", "NG(TO)"),
            ("ERROR", "NG(Err)"),
            ("NOT_FOUND", "NG(NF)"),
            ("SKIP", "SKIP"),
            ("DRY", "DRY"),
        ]:
            if counts.get(key):
                parts.append(f"{counts[key]} {label}")
        logger.info("  結果: %s", " / ".join(parts))
        logger.info("  合計時間: %.0fs (%.1f min)", total, total / 60)
        logger.info("  ログ: %s", self.jsonl_log_path)
        logger.info("=" * 78)

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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from arena.lib.paths import DATA_DIR, OUTPUT_DIR
from arena.lib.runtime_config import load_settings

DEFAULT_FILES = [
    ("adsb_airspy_decoder_metrics_1m.jsonl", "adsb_airspy_decoder_metrics_1m.jsonl", True),
    ("adsb_airspy_metrics_1m.jsonl", "adsb_airspy_metrics_1m.jsonl", False),
    ("dist_1m.jsonl", "dist_1m.jsonl", True),
    ("dist_pos_health_1m.jsonl", "dist_pos_health_1m.jsonl", True),
    ("dist_signal_stats_1m.jsonl", "dist_signal_stats_1m.jsonl", True),
]


@dataclass
class FileResult:
    local_name: str
    remote_path: str
    local_path: str
    status: str
    message: str
    returncode: int | None = None


def _to_wsl_path(p: Path) -> str:
    s = str(p)
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return s.replace("\\", "/")


def _load_sync_settings() -> dict[str, Any]:
    data = load_settings().data or {}
    sec = data.get("rpi_sync", {})
    return sec if isinstance(sec, dict) else {}


def _is_windows() -> bool:
    return os.name == "nt"


def _run_cmd(cmd: list[str], use_wsl: bool) -> subprocess.CompletedProcess:
    if not use_wsl:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = " ".join(shlex.quote(c) for c in cmd)
    return subprocess.run(
        ["wsl", "-e", "bash", "-lc", payload],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _build_ssh_transport(port: int, ssh_key: str, strict: str) -> str:
    parts = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", f"StrictHostKeyChecking={strict}"]
    if ssh_key:
        parts.extend(["-i", ssh_key])
    return " ".join(parts)


def _stat_remote(host: str, user: str, port: int, ssh_key: str, strict: str, remote_file: str, use_wsl: bool) -> tuple[bool, str]:
    target = f"{user}@{host}" if user else host
    # shellcheck-style single command for remote file existence check
    remote_cmd = f"test -e {shlex.quote(remote_file)} && echo OK || echo MISSING"
    cmd = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", f"StrictHostKeyChecking={strict}"]
    if ssh_key:
        cmd += ["-i", ssh_key]
    cmd += [target, remote_cmd]
    proc = _run_cmd(cmd, use_wsl=use_wsl)
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out.endswith("OK"):
        return True, "remote file exists"
    msg = (proc.stderr or out or "remote check failed").strip()
    return False, msg


def _remote_file_size(
    host: str,
    user: str,
    port: int,
    ssh_key: str,
    strict: str,
    remote_file: str,
    use_wsl: bool,
) -> tuple[int | None, str]:
    target = f"{user}@{host}" if user else host
    remote_cmd = f"stat -c %s {shlex.quote(remote_file)}"
    cmd = ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", f"StrictHostKeyChecking={strict}"]
    if ssh_key:
        cmd += ["-i", ssh_key]
    cmd += [target, remote_cmd]
    proc = _run_cmd(cmd, use_wsl=use_wsl)
    out = (proc.stdout or "").strip()
    if proc.returncode == 0:
        try:
            return int(out), "ok"
        except ValueError:
            return None, f"unexpected remote stat output: {out!r}"
    return None, (proc.stderr or out or "remote stat failed").strip()


def _backup_if_local_larger(*, local_file: Path, remote_size: int | None) -> str:
    if remote_size is None or not local_file.exists() or not local_file.is_file():
        return ""
    try:
        local_size = local_file.stat().st_size
    except OSError as exc:
        return f"local stat failed before rsync: {exc}"
    if local_size <= remote_size:
        return ""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = local_file.with_name(f"{local_file.name}.local-larger-{stamp}.bak")
    try:
        local_file.replace(backup)
    except OSError as exc:
        return f"local file larger than remote but backup failed: local={local_size} remote={remote_size} error={exc}"
    return f"local file larger than remote; backed up {local_file} -> {backup} before full re-sync (local={local_size} remote={remote_size})"


def _rsync_one(
    host: str,
    user: str,
    port: int,
    ssh_key: str,
    strict: str,
    remote_file: str,
    local_file: Path,
    use_wsl: bool,
    dry_run: bool,
) -> tuple[int, str]:
    target = f"{user}@{host}" if user else host
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_path = _to_wsl_path(local_file) if use_wsl else str(local_file)
    transport = _build_ssh_transport(port, ssh_key, strict)
    remote_size, remote_size_msg = _remote_file_size(
        host=host,
        user=user,
        port=port,
        ssh_key=ssh_key,
        strict=strict,
        remote_file=remote_file,
        use_wsl=use_wsl,
    )
    backup_msg = _backup_if_local_larger(local_file=local_file, remote_size=remote_size)
    cmd = [
        "rsync",
        "-av",
        "--append-verify",
        "--timeout=30",
        "-e",
        transport,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [f"{target}:{remote_file}", local_path]
    proc = _run_cmd(cmd, use_wsl=use_wsl)
    msg = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    notes = [note for note in (backup_msg, "" if remote_size is not None else f"remote size check skipped: {remote_size_msg}") if note]
    if notes:
        msg = "\n".join(notes + ([msg] if msg else []))
    return proc.returncode, msg


def _rsync_plao_dir(
    host: str,
    user: str,
    port: int,
    ssh_key: str,
    strict: str,
    remote_dir: str,
    local_dir: Path,
    use_wsl: bool,
    dry_run: bool,
) -> tuple[int, str]:
    target = f"{user}@{host}" if user else host
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = _to_wsl_path(local_dir) if use_wsl else str(local_dir)
    if not local_path.endswith("/"):
        local_path += "/"
    src = remote_dir.rstrip("/") + "/"
    transport = _build_ssh_transport(port, ssh_key, strict)
    cmd = [
        "rsync",
        "-av",
        "--append-verify",
        "--timeout=30",
        "--include=pos_*.jsonl",
        "--exclude=*",
        "-e",
        transport,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [f"{target}:{src}", local_path]
    proc = _run_cmd(cmd, use_wsl=use_wsl)
    msg = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode, msg


def build_parser() -> argparse.ArgumentParser:
    s = _load_sync_settings()
    ap = argparse.ArgumentParser(
        description="Sync ADS-B minute logs from Raspberry Pi to local data dir (append-safe)."
    )
    ap.add_argument("--host", default=str(s.get("host", "")))
    ap.add_argument("--user", default=str(s.get("user", "")))
    ap.add_argument("--port", type=int, default=int(s.get("ssh_port", 22)))
    ap.add_argument(
        "--remote-dir",
        default=str(s.get("remote_dir", "/path/to/adsb_monitor/data/logs")),
    )
    ap.add_argument(
        "--plao-remote-dir",
        default=str(s.get("plao_remote_dir", "/path/to/plao/data/logs/plao_pos")),
    )
    ap.add_argument(
        "--plao-local-dir",
        default=str(s.get("plao_local_dir", str(DATA_DIR / "plao_pos"))),
    )
    ap.add_argument("--skip-plao-sync", action="store_true")
    ap.add_argument("--ssh-key", default=str(s.get("ssh_key", "")))
    ap.add_argument(
        "--strict-host-key-checking",
        choices=["accept-new", "yes", "no"],
        default=str(s.get("strict_host_key_checking", "accept-new")),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-on-missing-remote", action="store_true")
    ap.add_argument(
        "--output-json",
        default=str(OUTPUT_DIR / "performance" / "rpi_log_sync_latest.json"),
    )
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if not args.host:
        print("[rpi_sync] --host is required (or set [rpi_sync].host in settings.toml)")
        return 2

    use_wsl = _is_windows()
    if use_wsl:
        probe = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "command -v rsync >/dev/null 2>&1; echo $?"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0 or not (probe.stdout or "").strip().endswith("0"):
            print("[rpi_sync] WSL rsync is not available.")
            return 2

    results: list[FileResult] = []
    remote_dir = args.remote_dir.rstrip("/")

    for local_name, remote_name, required in DEFAULT_FILES:
        local_path = DATA_DIR / local_name
        remote_path = f"{remote_dir}/{remote_name}"

        if args.dry_run:
            results.append(
                FileResult(
                    local_name=local_name,
                    remote_path=remote_path,
                    local_path=str(local_path),
                    status="ok",
                    message="dry-run (not executed)",
                    returncode=0,
                )
            )
            continue

        ok_remote, remote_msg = _stat_remote(
            host=args.host,
            user=args.user,
            port=args.port,
            ssh_key=args.ssh_key,
            strict=args.strict_host_key_checking,
            remote_file=remote_path,
            use_wsl=use_wsl,
        )
        if not ok_remote:
            status = "fail" if (required and args.fail_on_missing_remote) else "warn"
            results.append(
                FileResult(
                    local_name=local_name,
                    remote_path=remote_path,
                    local_path=str(local_path),
                    status=status,
                    message=f"remote missing/unreadable: {remote_msg}",
                    returncode=None,
                )
            )
            continue

        rc, msg = _rsync_one(
            host=args.host,
            user=args.user,
            port=args.port,
            ssh_key=args.ssh_key,
            strict=args.strict_host_key_checking,
            remote_file=remote_path,
            local_file=local_path,
            use_wsl=use_wsl,
            dry_run=args.dry_run,
        )
        status = "ok" if rc == 0 else "fail"
        results.append(
            FileResult(
                local_name=local_name,
                remote_path=remote_path,
                local_path=str(local_path),
                status=status,
                message=msg[-1200:],
                returncode=rc,
            )
        )

    if not args.skip_plao_sync:
        plao_remote = str(args.plao_remote_dir).rstrip("/")
        plao_local = Path(args.plao_local_dir)
        if args.dry_run:
            results.append(
                FileResult(
                    local_name="plao_pos/*.jsonl",
                    remote_path=f"{plao_remote}/",
                    local_path=str(plao_local),
                    status="ok",
                    message="dry-run (not executed)",
                    returncode=0,
                )
            )
        else:
            ok_remote, remote_msg = _stat_remote(
                host=args.host,
                user=args.user,
                port=args.port,
                ssh_key=args.ssh_key,
                strict=args.strict_host_key_checking,
                remote_file=plao_remote,
                use_wsl=use_wsl,
            )
            if not ok_remote:
                results.append(
                    FileResult(
                        local_name="plao_pos/*.jsonl",
                        remote_path=f"{plao_remote}/",
                        local_path=str(plao_local),
                        status="warn",
                        message=f"plao remote dir missing/unreadable: {remote_msg}",
                        returncode=None,
                    )
                )
            else:
                rc, msg = _rsync_plao_dir(
                    host=args.host,
                    user=args.user,
                    port=args.port,
                    ssh_key=args.ssh_key,
                    strict=args.strict_host_key_checking,
                    remote_dir=plao_remote,
                    local_dir=plao_local,
                    use_wsl=use_wsl,
                    dry_run=bool(args.dry_run),
                )
                results.append(
                    FileResult(
                        local_name="plao_pos/*.jsonl",
                        remote_path=f"{plao_remote}/",
                        local_path=str(plao_local),
                        status="ok" if rc == 0 else "fail",
                        message=msg[-1200:],
                        returncode=rc,
                    )
                )

    overall = "ok"
    if any(r.status == "fail" for r in results):
        overall = "fail"
    elif any(r.status == "warn" for r in results):
        overall = "warn"

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": args.host,
        "user": args.user,
        "remote_dir": args.remote_dir,
        "plao_remote_dir": args.plao_remote_dir,
        "plao_local_dir": args.plao_local_dir,
        "dry_run": bool(args.dry_run),
        "overall_status": overall,
        "results": [
            {
                "local_name": r.local_name,
                "remote_path": r.remote_path,
                "local_path": r.local_path,
                "status": r.status,
                "returncode": r.returncode,
                "message": r.message,
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[rpi_sync] status={overall} host={args.host}")
    for r in results:
        print(f"  - {r.status.upper():4} {r.local_name} <= {r.remote_path}")
    print(f"  wrote: {out}")

    return 0 if overall in ("ok", "warn") else 1


if __name__ == "__main__":
    raise SystemExit(main())

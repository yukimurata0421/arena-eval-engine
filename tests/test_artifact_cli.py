from __future__ import annotations

from arena import artifact_cli


def test_artifact_cli_parses_run_and_forwards_remainder(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_ensure() -> None:
        called["ensured"] = True

    def fake_legacy_main(argv) -> int:
        called["argv"] = list(argv)
        return 0

    monkeypatch.setattr(artifact_cli, "_ensure_repo_paths_on_syspath", fake_ensure)
    monkeypatch.setattr("scripts.tools.artifacts.cli.main", fake_legacy_main)

    rc = artifact_cli.main(["run", "--dry-run", "--", "--base", "x"])
    assert rc == 0
    assert called.get("ensured") is True
    assert called.get("argv") == ["--dry-run", "--base", "x"]


def test_artifact_cli_output_dir_sets_defaults(monkeypatch, tmp_path) -> None:
    called: dict[str, object] = {}

    monkeypatch.setattr(artifact_cli, "_ensure_repo_paths_on_syspath", lambda: None)
    monkeypatch.setattr("scripts.tools.artifacts.cli.main", lambda argv: called.__setitem__("argv", list(argv)) or 0)

    out_dir = tmp_path / "output"
    rc = artifact_cli.main(["run", "--output-dir", str(out_dir), "--dry-run"])
    assert rc == 0
    assert called["argv"][:4] == ["--base", str(out_dir.resolve()), "--out", str((out_dir / "payload").resolve())]


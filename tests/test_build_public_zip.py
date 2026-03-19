from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from scripts.tools import build_public_zip as bpz


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def test_build_public_zip_excludes_forbidden_entries(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    _write(repo / "src" / "app.py", "print('ok')\n")
    _write(repo / "README.md", "# test\n")
    _write(repo / ".git" / "config", "x\n")
    _write(repo / "output" / "secret.txt", "x\n")
    _write(repo / "scripts" / "tools" / "debug" / "tmp_probe.txt", "x\n")
    _write(repo / "scripts" / "structure", "x\n")
    _write(repo / "__pycache__" / "cache.pyc", "x\n")

    out_zip = repo / "public.zip"
    count, _size = bpz.build_zip(repo_root=repo, output_path=out_zip, includes=["."])
    assert count >= 2

    with ZipFile(out_zip, "r") as archive:
        names = set(archive.namelist())
    assert "src/app.py" in names
    assert "README.md" in names
    assert ".git/config" not in names
    assert "output/secret.txt" not in names
    assert "scripts/tools/debug/tmp_probe.txt" not in names
    assert "scripts/structure" not in names
    assert "__pycache__/cache.pyc" not in names


def test_build_public_zip_is_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "src" / "a.py", "a=1\n")
    _write(repo / "src" / "b.py", "b=2\n")
    _write(repo / "docs" / "note.md", "hello\n")

    zip_a = repo / "a.zip"
    zip_b = repo / "b.zip"

    bpz.build_zip(repo_root=repo, output_path=zip_a, includes=["src", "docs"])
    bpz.build_zip(repo_root=repo, output_path=zip_b, includes=["src", "docs"])

    assert zip_a.read_bytes() == zip_b.read_bytes()


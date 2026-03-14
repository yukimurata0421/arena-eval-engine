from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FileItem:
    rel_path: str
    abs_path: str
    ext: str
    size_bytes: int
    mtime_iso: str
    included: bool
    reason: str


@dataclass
class AIManifestRecord:
    relative_path: str
    category: str
    required_level: str
    exists: bool
    copied: bool
    size_bytes: int
    mtime: str
    status: str
    note: str
    source_path: str = ""
    copied_path: str = ""
    artifact_sha256: str = ""


@dataclass
class AIExportIntegrity:
    duplicate_source_skipped: int
    duplicate_output_paths: int
    copied_records: int
    missing_artifacts: int
    hash_mismatches: int
    validated_hash_entries: int
    passed: bool


@dataclass(frozen=True)
class AICandidateFile:
    logical_name: str
    relative_path: str
    expected_path: str
    priority: str
    reason: str
    what_it_enables: str
    rationale: str


@dataclass
class AICandidateStatus:
    logical_name: str
    relative_path: str
    expected_path: str
    exists: bool
    copied_to_export: bool
    priority: str
    note: str
    source_path: str
    reason: str
    what_it_enables: str
    rationale: str


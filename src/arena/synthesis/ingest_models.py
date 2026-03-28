from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arena.synthesis.validator import LoadedRecord

MODEL_CHOICES = ("claude", "gemini", "gpt", "grok")
SOURCE_STEM_RE = re.compile(r"^(?P<date>\d{8})(?:-(?P<slot>\d+))?$")


class RepairIngestError(ValueError):
    """Raised when repair pipeline could not produce DB-ingestable records."""


class RepairValidationError(ValueError):
    """Raised when repaired data failed validator checks."""


@dataclass(frozen=True, slots=True)
class IngestFileResult:
    source_path: Path
    model: str
    source_date: str | None
    records_ingested: int
    skipped_duplicate: bool
    warning_count: int
    notes: tuple[str, ...] = ()
    original_saved: int = 0
    repaired_saved: int = 0
    repairs_applied: int = 0


@dataclass(frozen=True, slots=True)
class IngestRawReport:
    scanned_files: int
    scanned_candidates: int
    ingested_files: int
    skipped_duplicates: int
    skipped_layout: int
    ingested_records: int
    failed_records: int
    warnings: tuple[str, ...]
    original_saved: int = 0
    repaired_saved: int = 0
    repairs_applied: int = 0
    warnings_count: int = 0
    repair_failed_files: int = 0
    validation_failed_files: int = 0

    @property
    def candidate_files(self) -> int:
        # Backward compatibility for existing callers/tests.
        return self.scanned_candidates


@dataclass(frozen=True, slots=True)
class RepairPipelineOutcome:
    records: tuple[LoadedRecord, ...]
    repaired_text: str
    syntax_repairs: tuple[str, ...]
    shape_repairs: tuple[str, ...]
    semantic_repairs: tuple[str, ...]
    warnings: tuple[str, ...]
    record_count_before: int
    record_count_after: int
    error_message: str | None = None

from __future__ import annotations

from arena import cli as arena_cli
from arena.artifacts import integrity as core_integrity
from arena.artifacts import replay as core_replay
from arena.artifacts import schema as core_schema
from arena.artifacts import selection as core_selection
from scripts.tools.artifacts import integrity as legacy_integrity
from scripts.tools.artifacts import replay as legacy_replay
from scripts.tools.artifacts import selection as legacy_selection


def test_legacy_core_modules_alias_public_core() -> None:
    assert legacy_selection is core_selection
    assert legacy_integrity is core_integrity
    assert legacy_replay is core_replay


def test_public_cli_artifacts_commands_resolve_to_core() -> None:
    assert arena_cli.verify_artifact_bundle.__module__.startswith("arena.artifacts.")
    assert arena_cli.replay_artifact_bundle.__module__.startswith("arena.artifacts.")


def test_state_names_keep_backward_compatible_values() -> None:
    required_states = {
        "missing_required",
        "missing_recommended",
        "excluded_by_rule",
        "duplicate_source_skipped",
        "copy_failed",
    }
    assert required_states.issubset(set(core_schema.STATE_NAMES))

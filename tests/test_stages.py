from __future__ import annotations

from arena.pipeline.stages import PipelineBuildOptions, Step, build_pipeline

# ── Helpers ──────────────────────────────────────────────────────────

_DEFAULT_ARGS = dict(dynamic_date="2026-02-14", full_mode=False, skip_plao=False, options=PipelineBuildOptions())


def _steps(**overrides) -> list[Step]:
    return build_pipeline(**{**_DEFAULT_ARGS, **overrides})


def _by_stage(steps: list[Step]) -> dict[int, list[Step]]:
    out: dict[int, list[Step]] = {}
    for s in steps:
        out.setdefault(s.stage, []).append(s)
    return out


# ── Snapshot: overall structure ──────────────────────────────────────

def test_total_step_count() -> None:
    assert len(_steps()) == 36


def test_step_count_per_stage() -> None:
    stages = _by_stage(_steps())
    expected = {1: 8, 2: 5, 3: 11, 4: 1, 5: 8, 6: 1, 7: 1, 8: 1}
    actual = {k: len(v) for k, v in stages.items()}
    assert actual == expected


def test_full_mode_adds_steps() -> None:
    normal = _steps(full_mode=False)
    full = _steps(full_mode=True)
    assert len(full) > len(normal)


def test_skip_plao_removes_stage7() -> None:
    normal = _steps(skip_plao=False)
    skipped = _steps(skip_plao=True)
    assert len(skipped) < len(normal)
    stage7_skipped = [s for s in skipped if s.stage == 7]
    assert stage7_skipped == []


# ── Script uniqueness ────────────────────────────────────────────────

def test_all_script_rels_unique() -> None:
    scripts = [s.script_rel for s in _steps()]
    assert len(scripts) == len(set(scripts)), f"Duplicate scripts: {[s for s in scripts if scripts.count(s) > 1]}"


# ── Critical / soft-fail properties ─────────────────────────────────

def test_critical_steps_are_in_stage1() -> None:
    for s in _steps():
        if s.critical:
            assert s.stage == 1, f"Critical step {s.script_rel} not in stage 1"


def test_critical_step_scripts() -> None:
    critical = sorted(s.script_rel for s in _steps() if s.critical)
    expected = sorted([
        "adsb/aggregators/adsb_aggregator.py",
        "adsb/aggregators/adsb_eval_pk_aggregator.py",
        "adsb/aggregators/adsb_csv_patcher.py",
        "adsb/aggregators/adsb_local_traffic_proxy_gen.py",
    ])
    assert critical == expected


def test_soft_fail_scripts() -> None:
    soft = sorted(s.script_rel for s in _steps() if s.soft_fail_on_error)
    expected = sorted([
        "adsb/ops/dist_1m_health_check.py",
        "adsb/data_fetch/get_opensky_traffic.py",
    ])
    assert soft == expected


# ── Signal aggregator (newly expanded) ──────────────────────────────

def test_signal_aggregator_properties() -> None:
    step = next(s for s in _steps() if "signal_stats_aggregator" in s.script_rel)
    assert step.stage == 1
    assert step.expected_outputs == ["adsb_signal_range_summary.csv"]
    assert step.always_run_when_skip_existing is True


# ── Expected outputs presence ────────────────────────────────────────

def test_every_step_has_script_rel() -> None:
    for s in _steps():
        assert s.script_rel, f"Empty script_rel in stage {s.stage}: {s.label}"


def test_stage_s1_06_uses_logical_data_output_path() -> None:
    steps = _steps()
    opensky = next(s for s in steps if s.script_rel == "adsb/data_fetch/get_opensky_traffic.py")
    assert opensky.expected_outputs == ["data://flight_data/airport_movements.csv"]


# ── Stage 1 wave invariants ──────────────────────────────────────────

def test_stage1_wave_assignments() -> None:
    steps = [s for s in _steps() if s.stage == 1]
    by_script = {s.script_rel: s for s in steps}
    expected = {
        "adsb/aggregators/adsb_aggregator.py": 1,
        "adsb/aggregators/adsb_eval_pk_aggregator.py": 2,
        "adsb/ops/dist_1m_health_check.py": 2,
        "adsb/aggregators/adsb_csv_patcher.py": 3,
        "signals/collectors/signal_stats_aggregator.py": 2,
        "adsb/data_fetch/get_opensky_traffic.py": 4,
        "adsb/aggregators/adsb_local_traffic_proxy_gen.py": 4,
        "adsb/analysis/time_resolved/adsb_time_resolved_aggregator.py": 4,
    }
    assert {k: by_script[k].wave for k in expected} == expected


# ── Depends-on invariants ────────────────────────────────────────────

def test_depends_on_outputs_contract() -> None:
    steps = _steps()
    by_script = {s.script_rel: s for s in steps}
    assert by_script["adsb/analysis/phase/adsb_phase_evaluator_v3.py"].depends_on_outputs == ["adsb_daily_summary.csv"]
    assert by_script["adsb/analysis/change_points/adsb_detect_multi_change_points.py"].depends_on_outputs == ["adsb_daily_summary_v2.csv"]
    assert by_script["adsb/analysis/change_points/adsb_detect_change_point.py"].depends_on_outputs == ["adsb_daily_summary_v2.csv"]
    assert by_script["adsb/analysis/change_points/adsb_daily_signature_change_points.py"].depends_on_outputs == ["adsb_daily_summary_v2.csv"]


# ── Step attribute invariants ────────────────────────────────────────

def test_timeout_positive() -> None:
    for s in _steps():
        assert s.timeout_s > 0, f"{s.script_rel} has non-positive timeout"


def test_stages_are_contiguous_from_1() -> None:
    stage_nums = sorted(set(s.stage for s in _steps()))
    assert stage_nums == list(range(1, max(stage_nums) + 1))


def test_error_code_base_is_sequential_per_stage() -> None:
    steps = _steps()
    by_stage: dict[int, list[Step]] = {}
    for s in steps:
        by_stage.setdefault(s.stage, []).append(s)
    for stage, sts in by_stage.items():
        codes = [s.error_code_base for s in sts]
        assert all(c.startswith(f"S{stage}-") for c in codes)
        assert codes == [f"S{stage}-{i:02d}" for i in range(1, len(sts) + 1)]


def test_full_mode_additional_steps_are_fixed() -> None:
    normal_scripts = {s.script_rel for s in _steps(full_mode=False)}
    full_scripts = {s.script_rel for s in _steps(full_mode=True)}
    added = full_scripts - normal_scripts
    assert added == {
        "adsb/heatmap/adsb_daily_heatmap.py",
        "adsb/analysis/change_points/adsb_multi_discovery.py",
    }


def test_skip_plao_does_not_change_other_stages() -> None:
    normal = _steps(skip_plao=False)
    skipped = _steps(skip_plao=True)
    normal_other = {s.script_rel for s in normal if s.stage != 7}
    skipped_all = {s.script_rel for s in skipped}
    assert normal_other == skipped_all


def test_stage2_main_scripts_fixed() -> None:
    steps = _steps()
    stage2 = [s.script_rel for s in steps if s.stage == 2]
    assert stage2 == [
        "adsb/analysis/reports/adsb_fringe_decoding_evaluator.py",
        "adsb/analysis/reports/adsb_polar_coverage_evaluator.py",
        "adsb/analysis/reports/adsb_vertical_profile_evaluator.py",
        "adsb/heatmap/adsb_heatmap_generator.py",
        "adsb/heatmap/adsb_daily_heatmap_alt.py",
    ]


def test_stage5_main_scripts_fixed() -> None:
    steps = _steps(full_mode=False)
    stage5 = [s.script_rel for s in steps if s.stage == 5]
    assert stage5 == [
        "adsb/analysis/gpu/adsb_bayesian_phase_cuda_eval.py",
        "adsb/analysis/bayesian/adsb_bayesian_dynamic_eval.py",
        "adsb/analysis/bayesian/adsb_bayesian_advi_eval.py",
        "adsb/analysis/change_points/adsb_detect_multi_change_points.py",
        "adsb/analysis/change_points/adsb_detect_change_point.py",
        "adsb/analysis/change_points/adsb_daily_signature_change_points.py",
        "adsb/analysis/gpu/adsb_cuda_evaluator.py",
        "adsb/analysis/gpu/adsb_cuda_processor.py",
    ]

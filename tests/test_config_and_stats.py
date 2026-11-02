"""Configuration validation, data files vs. baseline, and experiment statistics."""

from __future__ import annotations

import copy
import math

import pandas as pd
import pytest
from scipy import stats

from conftest import BASELINE_PATH, EXPERIMENTS_PATH, ROOT
from line_sim.config import ConfigError, apply_overrides, load_yaml, resolve_config, validate
from line_sim.experiments import (
    improvement_pct, load_matrix, mean_ci, paired_diff, scenario_overrides,
)


@pytest.fixture
def base():
    return load_yaml(BASELINE_PATH)


def test_baseline_valid(base):
    assert validate(base) == []


@pytest.mark.parametrize("path,value", [
    ("line.stations.TEST.times_s", {"P1": -1, "P2": 65, "P3": 90}),
    ("orders.counts", {"P1": 0, "P2": 0, "P3": 0}),
    ("orders.counts", {"P1": -5, "P2": 144, "P3": 96}),
    ("line.stations.ASM.machines", 0),
    ("products.P1", {"fail_rate": 1.2, "recover_prob": 0.85}),
    ("line.stations.PACK.input_buffer.capacity", 0),
    ("release.wip_limit", 0),
    ("stochastic.cv", -0.1),
])
def test_validate_blocks_bad_values(base, path, value):
    cfg = apply_overrides(base, {path: value})
    assert validate(cfg)
    with pytest.raises(ConfigError):
        resolve_config(cfg)


def test_unknown_override_key_rejected(base):
    with pytest.raises(ConfigError):
        apply_overrides(base, {"line.stations.TEST.speed": 2})
    with pytest.raises(ConfigError):
        apply_overrides(base, {"no_such_section.x": 1})
    assert apply_overrides(base, {"time_scale.ASM": 0.8})["time_scale"] == {"ASM": 0.8}
    assert base["time_scale"] == {}  # deep copy


def test_bad_route_and_scale(base):
    cfg = copy.deepcopy(base)
    cfg["line"]["stations"]["PACK"]["route"] = {"next": "NOWHERE"}
    cfg["time_scale"] = {"ASM": 0}
    errors = validate(cfg)
    assert any("route" in e for e in errors) and any("time_scale" in e for e in errors)


def test_product_specs_match_baseline(base):
    df = pd.read_csv(ROOT / "data" / "product_specs.csv", encoding="utf-8-sig")
    counts = base["orders"]["counts"]
    total = sum(counts.values())
    st = base["line"]["stations"]
    cols = {"prep_s": "PREP", "asm_s": "ASM", "test_s": "TEST", "rework_s": "REWORK", "pack_s": "PACK"}
    assert list(df.product_type) == list(counts)
    for r in df.itertuples(index=False):
        p = r.product_type
        assert r.order_count == counts[p]
        assert math.isclose(r.order_share, counts[p] / total)
        for col, sid in cols.items():
            assert getattr(r, col) == st[sid]["times_s"][p]
        assert r.fail_rate == base["products"][p]["fail_rate"]
        assert r.recover_prob == base["products"][p]["recover_prob"]


def test_assumptions_match_baseline(base):
    df = pd.read_csv(ROOT / "data" / "assumptions.csv", encoding="utf-8-sig", dtype=str)
    assert list(df.columns) == ["parameter", "value", "unit", "evidence_type", "source_or_reason", "limitation"]
    assert set(df.evidence_type) <= {"assumed", "design_choice", "derived"}
    v = dict(zip(df.parameter, df.value))
    st = base["line"]["stations"]
    for sid in st:
        assert float(v[f"machines_{sid}"]) == st[sid]["machines"]
        bid = st[sid]["input_buffer"]["id"]
        assert float(v[f"buffer_capacity_{bid}"]) == st[sid]["input_buffer"]["capacity"]
        for p, t in st[sid]["times_s"].items():
            assert float(v[f"{sid.lower()}_time_{p}"]) == t
    for a, row in st["ASM"]["setup_matrix_s"].items():
        for b, t in row.items():
            if a != b:
                assert float(v[f"asm_setup_{a}_to_{b}"]) == t
    for p, pc in base["products"].items():
        assert float(v[f"fail_rate_{p}"]) == pc["fail_rate"]
        assert float(v[f"recover_prob_{p}"]) == pc["recover_prob"]
    assert float(v["release_min_interval"]) == base["release"]["min_interval_s"]
    assert float(v["wip_limit"]) == base["release"]["wip_limit"]
    assert float(v["processing_time_cv"]) == base["stochastic"]["cv"]
    assert float(v["time_limit"]) == base["run"]["time_limit_s"]


def test_mean_ci_hand_calculation():
    vals = [2.0, 4.0, 6.0, 8.0]
    ci = mean_ci(vals)
    sd = math.sqrt(((3 ** 2) + 1 + 1 + 9) / 3)  # 2.5819888974716
    hw = stats.t.ppf(0.975, 3) * sd / 2
    assert ci["n"] == 4 and ci["mean"] == 5.0
    assert math.isclose(ci["sd"], sd)
    assert math.isclose(ci["ci_low"], 5.0 - hw) and math.isclose(ci["ci_high"], 5.0 + hw)
    assert math.isclose(stats.t.ppf(0.975, 3), 3.182446305284263)


def test_mean_ci_single_value():
    ci = mean_ci([3.0])
    assert ci["mean"] == 3.0 and ci["ci_low"] is None and ci["ci_high"] is None and ci["sd"] is None


def test_paired_diff_pairs_by_seed():
    cand = {100: 12.0, 101: 15.0, 102: None, 103: 9.0}
    base = {100: 10.0, 101: 11.0, 102: 8.0, 104: 7.0}
    out = paired_diff(cand, base)
    assert out["n_pairs"] == 2
    assert out["dropped_seeds"] == [102, 103, 104]
    assert out["mean_d"] == 3.0
    assert out["base_mean"] == 10.5 and out["cand_mean"] == 13.5


def test_improvement_pct():
    assert improvement_pct(0.0, 5.0) is None
    assert improvement_pct(None, 5.0) is None
    assert improvement_pct(200.0, 210.0) == 5.0


def test_combination_rules():
    m = load_matrix(EXPERIMENTS_PATH)
    m["combinations"] = {"C1": {"components": ["S1", "S3"], "rationale": "TEST 증설 후 ASM 제약 확인"}}
    assert scenario_overrides(m, "C1") == {"line.stations.TEST.machines": 2, "time_scale.ASM": 0.8}
    m["scenarios"]["SX"] = {"overrides": {"line.stations.TEST.machines": 3}}
    m["combinations"]["C2"] = {"components": ["S1", "SX"], "rationale": "충돌 확인"}
    with pytest.raises(ConfigError):
        scenario_overrides(m, "C2")
    m["combinations"]["C2"] = {"components": ["S1", "S5"], "rationale": "x"}
    m["combinations"]["C3"] = {"components": ["S3", "S5"], "rationale": "x"}
    with pytest.raises(ConfigError):
        scenario_overrides(m, "C1")
    m["combinations"] = {"C1": {"components": ["S1", "S3"], "rationale": ""}}
    with pytest.raises(ConfigError):
        scenario_overrides(m, "C1")


def test_matrix_scenarios_resolve():
    from line_sim.experiments import resolve_run_config
    m = load_matrix(EXPERIMENTS_PATH)
    for sc in m["scenarios"]:
        for cond in m["conditions"]:
            resolve_run_config(m, sc, cond)


def test_window_and_time_limit_validation(base):
    cfg = apply_overrides(base, {"run.time_limit_s": 20000})
    assert any("time_limit" in e for e in validate(cfg))
    for win in ([-1, 28800], [28800, 28800], [100, 50]):
        assert validate(apply_overrides(base, {"run.observation_window_s": win}))


def test_route_cycle_and_single_inspect_rework(base):
    cfg = copy.deepcopy(base)
    cfg["line"]["stations"]["PACK"]["route"] = {"next": "PREP"}
    assert any("순환" in e for e in validate(cfg))
    cfg = copy.deepcopy(base)
    cfg["line"]["stations"]["PACK"]["route"] = {"type": "inspect", "on_pass": "SHIP", "on_fail": "REWORK"}
    assert any("inspect" in e for e in validate(cfg))
    cfg = copy.deepcopy(base)
    cfg["line"]["stations"]["REWORK"]["route"] = {"type": "rework", "on_pass": "TEST", "on_fail": "SCRAP"}
    assert any("순환" in e for e in validate(cfg))


def test_load_errors_are_config_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(bad)
    partial = tmp_path / "m.yaml"
    partial.write_text("base_config: x.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_matrix(partial)


def test_seed_and_scenario_handling():
    from line_sim.experiments import expand_runs, parse_seeds
    assert parse_seeds("1000-1002,1001") == [1000, 1001, 1002]
    for text in ("abc", "5-1", ","):
        with pytest.raises(ConfigError):
            parse_seeds(text)
    m = load_matrix(EXPERIMENTS_PATH)
    runs = expand_runs(m, "discovery", scenarios=["B0", "S1", "B0"], seeds=[100, 100, 101])
    assert len(runs) == 4
    with pytest.raises(ConfigError):
        expand_runs(m, "discovery", seeds=[])
    with pytest.raises(ConfigError):
        expand_runs(m, "confirmation", candidate="B0")


def _small_matrix(tmp_path):
    import yaml
    m = load_yaml(EXPERIMENTS_PATH)
    m["base_config"] = str(BASELINE_PATH)
    m["phases"]["discovery"] = {"seeds": {"start": 100, "count": 2}, "scenarios": ["B0", "S1"],
                                "conditions": ["base"]}
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_run_experiment_smoke(tmp_path):
    from line_sim.experiments import run_experiment
    exp_dir, n_failed = run_experiment(_small_matrix(tmp_path), "discovery", experiment_id="smoke",
                                       results_root=tmp_path / "results", verbose=False)
    assert n_failed == 0
    index = pd.read_csv(exp_dir / "runs_index.csv", encoding="utf-8-sig")
    assert len(index) == 4 and set(index.status) == {"complete"}
    summary = pd.read_csv(exp_dir / "summary.csv", encoding="utf-8-sig")
    assert set(summary.scenario) == {"B0", "S1"} and (summary.n_failed == 0).all()
    paired = pd.read_csv(exp_dir / "paired.csv", encoding="utf-8-sig")
    gs = paired[paired.metric == "shift.good_shipped"]
    assert len(gs) == 1 and gs.n_pairs.iloc[0] == 2
    assert not pd.read_csv(exp_dir / "summary_by_product.csv", encoding="utf-8-sig").empty
    with pytest.raises(FileExistsError):
        run_experiment(_small_matrix(tmp_path), "discovery", experiment_id="smoke",
                       results_root=tmp_path / "results", verbose=False)


def test_run_experiment_without_pairs(tmp_path):
    from line_sim.experiments import main, read_csv_or_empty
    code = main(["--matrix", str(_small_matrix(tmp_path)), "--phase", "discovery", "--scenarios", "B0",
                 "--seeds", "100", "--experiment-id", "solo", "--results-root", str(tmp_path / "results")])
    assert code == 0
    paired = read_csv_or_empty(tmp_path / "results" / "solo" / "paired.csv")
    assert paired.empty

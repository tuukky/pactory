"""2b 실험 도구: MSER-5, 단일 변경 검증, 후보 선택 규칙(§9.2), 직렬 = 병렬."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from v2_helpers import rev_a, short
from factory_sim.calibration import check_phenomenon_only, summarize_calibration
from factory_sim.config import ConfigError
from factory_sim.experiments import (
    BASE, NO_CLEAR_DIFF, evaluate_targets, expand_runs, freeze_revision, load_matrix, paired_table,
    run_experiment, scenarios, select_candidate, validate_candidate,
)
from factory_sim.warmup import decide_warmup_h, hourly_good, mser5


# ---------------------------------------------------------------------------- MSER-5
def test_mser5_detects_initial_transient():
    # 처음 20시간은 0, 이후 10 → 5h 묶음 4개가 편향 → d* = 4 묶음 = 20h
    x = np.r_[np.zeros(20), np.full(100, 10.0)]
    r = mser5(x)
    assert r["d_batches"] == 4 and r["d_hours"] == 20 and r["n_batches"] == 24


def test_mser5_stationary_is_zero_and_curve_hand_calc():
    r = mser5([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])  # 묶음 평균 [1, 3], d ∈ {0, 1}
    assert r["mser_curve"][0] == pytest.approx(((1 - 2) ** 2 + (3 - 2) ** 2) / 4)
    assert r["mser_curve"][1] == pytest.approx(0.0)
    assert r["d_batches"] == 1
    assert mser5(np.full(50, 7.0))["d_batches"] == 0
    with pytest.raises(ValueError):
        mser5([1, 2, 3])


def test_warmup_decision_rule():
    assert decide_warmup_h([0, 20, 75]) == {"max_d_hours": 75.0, "warmup_h": 80.0, "changed": False}
    assert decide_warmup_h([81]) == {"max_d_hours": 81.0, "warmup_h": 88.0, "changed": True}


def test_hourly_good_bins():
    o = pd.DataFrame({"outcome": ["shipped", "shipped", "scrapped", "shipped", None],
                      "terminal_s": [10.0, 3599.9, 100.0, 7200.0, None]})
    assert hourly_good(o, 3 * 3600).tolist() == [2.0, 0.0, 1.0]


# ---------------------------------------------------------------------------- 보정 규칙
def test_calibration_only_phenomenon_params():
    assert check_phenomenon_only({"stations.R1.setup_group_change_s": 40, "failures.mtbf_scale": 0.9}) == []
    assert check_phenomenon_only({"stations.R1.machines": 4})
    assert check_phenomenon_only({"housing.reorder_point": 800})
    assert check_phenomenon_only({"products.groups.A.share": 0.4})
    s = summarize_calibration([{"down_ratio_load": 0.07, "setup_ratio.R1": 0.2, "setup_ratio.S3": 0.1,
                                "eol_first_pass_yield": 0.95}])
    assert s.set_index("metric")["in_range"].to_dict() == {"down_ratio_load": True, "setup_ratio.R1": False,
                                                           "setup_ratio.S3": True, "eol_first_pass_yield": True}


# ---------------------------------------------------------------------------- 단일 변경
def test_validate_candidate_rules():
    base = rev_a()
    ok = lambda c: validate_candidate(base, "X", c) == []  # noqa: E731
    assert ok({"resource_class": "rule", "overrides": {"housing.reorder_point": 800}})
    assert not ok({"resource_class": "rule", "overrides": {"housing.reorder_point": 800, "release.kit_slots": 40}})
    # 재배분: 총 슬롯 불변 (6 + 6 = 12)
    assert ok({"resource_class": "buffer_realloc", "overrides": {"stations.S5.input_buffer.capacity": 10,
                                                                 "stations.S4.input_buffer.capacity": 2}})
    assert not ok({"resource_class": "buffer_realloc", "overrides": {"stations.S5.input_buffer.capacity": 10,
                                                                     "stations.S4.input_buffer.capacity": 6}})
    assert ok({"resource_class": "skill_add", "overrides": {"operators.skills.OP2.skills": ["R3", "R2", "R1"]}})
    assert not ok({"resource_class": "skill_add", "overrides": {"agv.count": 4}})
    assert ok({"resource_class": "machine_add", "overrides": {"stations.S5.machines": 2,
                                                              "layout.nodes.S5_2": [38, 24]}})
    assert not ok({"resource_class": "machine_add", "overrides": {"stations.S5.machines": 2,
                                                                  "stations.R1.machines": 4}})


# ---------------------------------------------------------------------------- 선택 규칙
def _paired(rows):
    cols = ["scenario", "condition", "metric", "mean_d", "ci_low", "ci_high", "verdict"]
    return pd.DataFrame(rows, columns=cols)


P, O, L = "factory.demand_fulfil_ratio", "factory.otd_rate", "factory.lead_time_cohort_mean_s"


def test_select_candidate_rank_then_gain():
    rows = [
        # H1: 개선, rank 3 (AGV)
        ("H1", "base", P, 0.10, 0.08, 0.12, "개선"), ("H1", "base", O, 0.1, 0.05, 0.15, "개선"),
        # H2: 개선이 작지만 rank 0 → 채택
        ("H2", "base", P, 0.02, 0.01, 0.03, "개선"), ("H2", "base", O, 0.0, -0.01, 0.01, NO_CLEAR_DIFF),
        # H3: rank 0, 개선 크지만 리드타임 악화 → 부적격
        ("H3", "base", P, 0.20, 0.15, 0.25, "개선"), ("H3", "base", L, 500, 100, 900, "악화"),
        # H4: rank 0, CI가 0 포함 → 부적격
        ("H4", "base", P, 0.05, -0.01, 0.11, NO_CLEAR_DIFF),
        # H5: rank 0, H2보다 개선 큼 → 동률 rank에서 gain 큰 쪽
        ("H5", "base", P, 0.04, 0.02, 0.06, "개선"),
    ]
    ranks = {"H1": 3, "H2": 0, "H3": 0, "H4": 0, "H5": 0}
    res = select_candidate(_paired(rows), P, "base", ranks)
    t = res["table"].set_index("scenario")
    assert res["chosen"] == "H5"
    assert not t.loc["H3", "eligible"] and t.loc["H3", "worsened_targets"] == L
    assert not t.loc["H4", "eligible"]
    # 리드타임(낮을수록 좋음) primary: 음의 차이가 개선
    rows2 = [("A1", "base", L, -900, -1200, -600, "개선"), ("A2", "base", L, -2000, -2500, -1500, "개선")]
    assert select_candidate(_paired(rows2), L, "base", {"A1": 2, "A2": 2})["chosen"] == "A2"
    assert select_candidate(_paired(rows2[:0]), L, "base", {})["chosen"] is None
    with pytest.raises(ConfigError):
        select_candidate(_paired(rows), "factory.nope", "base", ranks)


def test_paired_verdict_direction():
    vals = pd.DataFrame([
        {"scenario": BASE, "condition": "base", "seed": s, "metric": L, "value": 1000.0 + s} for s in range(5)
    ] + [
        {"scenario": "H1", "condition": "base", "seed": s, "metric": L, "value": 900.0 + s + 0.1 * s * s}
        for s in range(5)
    ])
    p = paired_table(vals, {"scenarios": {"H1": {"resource_class": "rule"}}})
    r = p[p.metric == L].iloc[0]
    assert r["n_pairs"] == 5 and r["mean_d"] < 0 and r["verdict"] == "개선"


def test_evaluate_targets_mean_and_ci():
    summ = pd.DataFrame([
        {"scenario": "X", "condition": "base", "metric": P, "mean": 0.985, "ci_low": 0.975, "ci_high": 0.995},
        {"scenario": "X", "condition": "base", "metric": O, "mean": 0.97, "ci_low": 0.96, "ci_high": 0.98},
        {"scenario": "X", "condition": "base", "metric": L, "mean": 50000, "ci_low": 48000, "ci_high": 52000},
    ])
    t = evaluate_targets(summ, {"demand_fulfil_ratio_min": 0.98, "otd_min": 0.95,
                                "lead_time_mean_max_s": 57600}).iloc[0]
    assert t["all_ok"] and not t["all_ci_ok"] and not t["throughput_ci_ok"] and t["leadtime_ci_ok"]


# ---------------------------------------------------------------------------- 행렬·실행
def _tiny_matrix(tmp_path, candidates=None, combinations=None):
    rev = short(rev_a(), warmup_h=4, observation_h=12)
    (tmp_path / "rev_T.yaml").write_text(yaml.safe_dump(rev, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m = {"results_root": str(tmp_path / "res"), "revisions": {"T": "rev_T.yaml"},
         "resource_classes": {"rule": 0, "buffer_realloc": 0, "agv_add": 3},
         "studies": {"s1": {"base_rev": "T", "primary_metric": P,
                            "candidates": candidates if candidates is not None else
                            {"H1": {"resource_class": "rule", "description": "s 상향",
                                    "overrides": {"housing.reorder_point": 800}}},
                            "combinations": combinations or {}}},
         "conditions": {"base": {"overrides": {}}, "aging": {"overrides": {"failures.mtbf_scale": 0.5}}},
         "phases": {"explore": {"seeds": {"start": 1, "count": 2}, "conditions": ["base"]}}}
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(m, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_matrix_rejects_bad_candidates(tmp_path):
    bad = {"H1": {"resource_class": "rule", "overrides": {"housing.reorder_point": 800, "agv.count": 4}}}
    with pytest.raises(ConfigError):
        scenarios(load_matrix(_tiny_matrix(tmp_path, bad)), "s1")
    combos = {"C1": {"components": ["H1", "H2"], "rationale": ""}}
    cands = {"H1": {"resource_class": "rule", "overrides": {"housing.reorder_point": 800}},
             "H2": {"resource_class": "agv_add", "overrides": {"agv.count": 4}}}
    with pytest.raises(ConfigError):
        scenarios(load_matrix(_tiny_matrix(tmp_path, cands, combos)), "s1")
    combos["C1"]["rationale"] = "H1·H2 각각 개선"
    sc = scenarios(load_matrix(_tiny_matrix(tmp_path, cands, combos)), "s1")
    assert sc["C1"]["overrides"] == {"housing.reorder_point": 800, "agv.count": 4} and sc["C1"]["rank"] == 3
    runs = expand_runs(load_matrix(_tiny_matrix(tmp_path, cands, combos)), "s1", "explore", seeds=[1, 1, 2])
    assert len(runs) == 4 * 2


def test_serial_equals_parallel_and_freeze(tmp_path):
    mp = _tiny_matrix(tmp_path)
    d1, f1 = run_experiment(mp, "s1", "explore", workers=1, experiment_id="serial",
                            results_root=tmp_path / "r", verbose=False)
    d2, f2 = run_experiment(mp, "s1", "explore", workers=2, experiment_id="parallel",
                            results_root=tmp_path / "r", verbose=False)
    assert f1 == f2 == 0
    for rel in pd.read_csv(d1 / "runs_index.csv", encoding="utf-8-sig")["run_dir"]:
        m1 = json.loads((d1 / rel / "metrics.json").read_text(encoding="utf-8"))
        m2 = json.loads((d2 / rel / "metrics.json").read_text(encoding="utf-8"))
        assert m1 == m2
    s1 = pd.read_csv(d1 / "summary.csv", encoding="utf-8-sig")
    s2 = pd.read_csv(d2 / "summary.csv", encoding="utf-8-sig")
    pd.testing.assert_frame_equal(s1, s2)
    paired = pd.read_csv(d1 / "paired.csv", encoding="utf-8-sig")
    assert set(paired.scenario) == {"H1"} and (paired[paired.metric == P].n_pairs == 2).all()
    with pytest.raises(FileExistsError):
        run_experiment(mp, "s1", "explore", workers=1, experiment_id="serial", results_root=tmp_path / "r",
                       verbose=False)
    out, diff = freeze_revision(mp, "s1", "H1", "U", experiment_id="serial")
    assert diff == {"housing.reorder_point": (300, 800)}
    frozen = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert frozen["revision"] == "U" and frozen["frozen_from"]["candidate"] == "H1"
    with pytest.raises(FileExistsError):
        freeze_revision(mp, "s1", "H1", "U")

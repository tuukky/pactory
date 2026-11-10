"""PLAN_v5 S3 — VP 셀(kind: vision_pick) 검증 (V-S3b, 분포·민감도 규칙, 재현성)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory_sim.config import resolve_config, validate
from factory_sim.engine import run_simulation
from factory_sim.vision_pick import load_dist

ROOT = Path(__file__).resolve().parents[2]
VP_CFG = ROOT / "configs" / "v5" / "rev_C_vp.yaml"
SHORT = {"run.warmup_s": 20000, "run.observation_s": 60000, "demand.rate_per_h": 20.7}


def _dist(tmp_path, **kw):
    d = {"source": "test", "vision_time_s": [1.0], "robot_time_s": {"success": [10.0], "fail": [4.0]},
         "p_reject": 0.2, "p_fail": 0.1}
    d.update(kw)
    p = tmp_path / "d.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_rev_c_vp_config_valid():
    cfg = resolve_config(VP_CFG, {"demand.rate_per_h": 20.7})
    assert validate(cfg) == []
    assert cfg["routing"]["parts"]["HOUSING"]["path"] == ["VP"]
    assert cfg["stations"]["VP"]["kind"] == "vision_pick"


def test_attempt_fail_rate_keeps_reject_share(tmp_path):
    d = load_dist({"dist_file": _dist(tmp_path), "attempt_fail_rate": 0.06})
    assert d["p_reject"] + d["p_fail"] == pytest.approx(0.06)
    assert d["p_reject"] / (d["p_reject"] + d["p_fail"]) == pytest.approx(2 / 3)


def test_invalid_probability_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_dist({"dist_file": _dist(tmp_path, p_reject=0.8, p_fail=0.5)})


def test_vp_mass_balance_and_invariants(tmp_path):
    """V-S3b: 물량 수지 — 하우징은 VP에서 사라지지 않는다 (수동 처리 포함, 폐기 없음). 불변식 검사 켬."""
    cfg = resolve_config(VP_CFG, {**SHORT, "stations.VP.vision_pick.dist_file": _dist(tmp_path, p_reject=0.5, p_fail=0.3)})
    r = run_simulation(cfg, 11, check_invariants=True)
    vp = r.stats["vision_pick"]["VP"]
    released = r.stats["released"]
    assert vp["housings"] <= released
    assert released - vp["housings"] <= 10        # 남은 재공(VP 입력·처리 중)만큼만 차이
    assert vp["manual"] > 0 and vp["attempts"] >= vp["housings"]


def test_vp_attempt_accounting(tmp_path):
    """시도 수 = 성공 시도 + 거부 + 파지 실패, 수동 처리는 N회 모두 실패한 하우징."""
    cfg = resolve_config(VP_CFG, {**SHORT, "stations.VP.vision_pick.dist_file": _dist(tmp_path, p_reject=0.4, p_fail=0.2)})
    r = run_simulation(cfg, 12)
    vp = r.stats["vision_pick"]["VP"]
    successes = vp["housings"] - vp["manual"]
    assert vp["attempts"] == successes + vp["rejects"] + vp["grasp_fails"]
    # 하우징마다 최대 3회
    assert vp["attempts"] <= 3 * vp["housings"]


def test_vp_reproducible(tmp_path):
    cfg = resolve_config(VP_CFG, {**SHORT, "stations.VP.vision_pick.dist_file": _dist(tmp_path)})
    a = run_simulation(cfg, 5)
    b = run_simulation(cfg, 5)
    assert a.stats["vision_pick"] == b.stats["vision_pick"]
    assert a.orders_frame().equals(b.orders_frame())


def test_crn_attempt_draws_independent_of_fail_rate(tmp_path):
    """민감도 짝 비교: 실패율을 올리면 수동 처리가 줄지 않는다 (같은 균등난수)."""
    p = _dist(tmp_path)
    lo = run_simulation(resolve_config(VP_CFG, {**SHORT, "stations.VP.vision_pick.dist_file": p,
                                                "stations.VP.vision_pick.attempt_fail_rate": 0.1}), 7)
    hi = run_simulation(resolve_config(VP_CFG, {**SHORT, "stations.VP.vision_pick.dist_file": p,
                                                "stations.VP.vision_pick.attempt_fail_rate": 0.4}), 7)
    assert hi.stats["vision_pick"]["VP"]["manual"] >= lo.stats["vision_pick"]["VP"]["manual"]
    assert hi.stats["vision_pick"]["VP"]["attempts"] >= lo.stats["vision_pick"]["VP"]["attempts"]


def test_v2_config_has_no_vp_stats():
    """v2 설정은 VP가 없으면 stats에 vision_pick 키가 없다 (v2 결과 파일 형식 유지)."""
    cfg = resolve_config(ROOT / "configs" / "v2" / "rev_C.yaml", {"run.warmup_s": 3600, "run.observation_s": 7200})
    r = run_simulation(cfg, 1)
    assert "vision_pick" not in r.stats

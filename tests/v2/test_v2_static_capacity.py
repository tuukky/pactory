"""정적 부하 계산(§5) 손계산 일치."""

from __future__ import annotations

import pytest

from v2_helpers import REV_A_PATH
from factory_sim.config import resolve_config
from factory_sim.static_capacity import compute, main, mean_station_time


@pytest.fixture(scope="module")
def res():
    return compute(resolve_config(REV_A_PATH))


def test_station_capacity_hand_calc(res):
    cfg = resolve_config(REV_A_PATH)
    assert mean_station_time(cfg, "R1") == pytest.approx(0.5 * 170 + 0.3 * 200 + 0.2 * 240)  # SKU 계수 평균 1
    st = res["stations"].set_index("station")
    p_change = 1 - (0.5 ** 2 + 0.3 ** 2 + 0.2 ** 2)
    assert res["group_change_prob"] == pytest.approx(p_change)
    assert st.loc["R1", "cap_pure_per_h"] == pytest.approx(3 * 3600 / 193)
    s_r1 = cfg["stations"]["R1"]["setup_group_change_s"]  # 보정 동결값 37.5 (calibration.md)
    assert s_r1 == 37.5
    assert st.loc["R1", "cap_setup_per_h"] == pytest.approx(3 * 3600 / (193 + p_change * s_r1))
    ff = 1 + 2100 / 14400
    assert st.loc["R1", "cap_fail_per_h"] == pytest.approx(3 * 3600 / (193 * ff + p_change * s_r1))
    assert st.loc["S5", "cap_pure_per_h"] == pytest.approx(3600 * 24 / 1800)
    fail = 0.5 * 0.03 + 0.3 * 0.05 + 0.2 * 0.08
    assert st.loc["EOL", "visits_per_order"] == pytest.approx(1 + fail)
    # 오븐은 실효 능력(입력 버퍼 제약 배치)으로 순위를 매긴다(C1): 이월 min(6+2, 18/h×0.5h=9)=8,
    # 형성 900s 동안 +4.5 → 배치 12.5, 주기 2700s → 16.67/h, 고장 반영 ÷(1+3600/43200)
    assert st.loc["S5", "cap_effective_per_h"] == pytest.approx(12.5 * 3600 / 2700 / (1 + 3600 / 43200))
    assert list(st.sort_values("rank").index[:3]) == ["S5", "R1", "S3"]
    assert st.loc["R1", "cap_effective_per_h"] == st.loc["R1", "cap_fail_per_h"]


def test_housing_supply_cap(res):
    h = res["housing"]
    # 리드타임 32h 수요 576 > s=300 → 주기 = L + (Q−s)/D = 32 + 300/18 h, 공급 = 600 / 주기
    assert h["supply_cap_per_h"] == pytest.approx(600 / (32 + 300 / 18))


def test_operator_and_agv_load(res):
    o = res["operators"]
    assert o["operators"] == 8 and 0 < o["util_at_demand"] < 1
    a = res["agv"]
    assert a["agv_needed_batch1"] == pytest.approx(4 * a["agv_needed_batch4"])
    assert a["availability"] == pytest.approx(14400 / 16200)


def test_cli_prints_table(capsys):
    assert main(["--config", str(REV_A_PATH)]) == 0
    out = capsys.readouterr().out
    assert "| 1 | S5 |" in out and "만재 24 가정" in out and "하우징" in out


def _over(**kw):
    from factory_sim.config import apply_overrides
    return compute(resolve_config(apply_overrides(resolve_config(REV_A_PATH), kw)))


def test_zero_fail_rate_no_division_error():
    r = _over(**{f"products.groups.{g}.fail_rate": 0.0 for g in "ABC"})
    st = r["stations"].set_index("station")
    assert st.loc["RW", "visits_per_order"] == 0 and st.loc["RW", "cap_pure_per_h"] == float("inf")
    assert st.loc["RW", "util_at_demand"] == 0
    lanes = r["agv"]["lanes"].set_index("lane")
    assert lanes.loc["EOL→PK", "trips_per_order"] == 1.0 and lanes.loc["EOL→RW", "trips_per_order"] == 0.0


def test_no_rework_means_no_rw_visits():
    r = _over(**{"routing.inspect.max_rework": 0})
    st = r["stations"].set_index("station")
    fail = 0.5 * 0.03 + 0.3 * 0.05 + 0.2 * 0.08
    assert st.loc["RW", "visits_per_order"] == 0 and st.loc["EOL", "visits_per_order"] == 1.0
    lanes = r["agv"]["lanes"].set_index("lane")
    assert lanes.loc["EOL→PK", "trips_per_order"] == pytest.approx(1 - fail)
    base = compute(resolve_config(REV_A_PATH))["operators"]["operator_s_per_order"]
    rw_op = fail * (0.5 * 300 + 0.3 * 360 + 0.2 * 450)
    assert r["operators"]["operator_s_per_order"] == pytest.approx(base - rw_op)


def test_recover_prob_drives_pack_lane():
    r = _over(**{f"products.groups.{g}.recover_prob": 0.5 for g in "ABC"})
    fail = 0.5 * 0.03 + 0.3 * 0.05 + 0.2 * 0.08
    lanes = r["agv"]["lanes"].set_index("lane")
    assert lanes.loc["EOL→PK", "trips_per_order"] == pytest.approx(1 - fail + 0.5 * fail)
    assert lanes.loc["PK→DOCK", "trips_per_order"] == pytest.approx(1 - fail + 0.5 * fail)

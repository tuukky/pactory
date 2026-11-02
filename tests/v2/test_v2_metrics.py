"""지표 정의: 반열린 관측창 [w0, w1), 도착 코호트 리드타임(절단 하한), 리드타임 목표 판정,
출력 버퍼 필수 검증, env_info 재사용, AGV 같은 정차 추가 적재."""

from __future__ import annotations

import numpy as np
import pytest

from v2_helpers import explicit_orders, motor_config, rev_a, single_line
from factory_sim.config import ConfigError, apply_overrides, resolve_config
from factory_sim.engine import run_simulation
from factory_sim.metrics import compute_metrics


def _line_run(target_s=95.0):
    # M 30초, 5개 0초 도착 → 출하 30·60·90·120·150 예정. 관측창 [0, 90) = 호라이즌 90에서 절단.
    # 90에 끝나는 3번은 경계 이벤트까지 처리되어 출하, 4·5번은 미종결(절단).
    cfg = single_line({"M": 30.0}, orders=explicit_orders(["A1"] * 5), observation_s=90.0)
    cfg["targets"] = {"lead_time_mean_max_s": target_s}
    return run_simulation(cfg, seed=1)


def test_half_open_window_excludes_w1():
    r = _line_run()
    assert r.status == "complete_at_horizon" and r.sim_end_s == 90.0
    f = compute_metrics(r)["factory"]
    assert f["good_in_window"] == 2  # 90에 출하된 것은 [0, 90) 밖
    assert f["lead_time"]["n"] == 2 and f["lead_time"]["mean_s"] == 45.0


def test_arrival_cohort_lead_time_and_target():
    # 코호트 = [0, 90) 도착 5건: 출하 30·60·90, 미종결 2건은 sim_end − 도착 = 90 (하한)
    m = compute_metrics(_line_run(95.0))
    c = m["factory"]["lead_time_arrival_cohort"]
    assert c == {"n": 5, "n_censored": 2, "mean_lower_bound_s": 72.0,
                 "p90_lower_bound_s": float(np.percentile([30, 60, 90, 90, 90], 90))}
    assert m["targets"]["leadtime_ok"] is True
    # 목표 70s: 출하분 평균(45s)이면 통과지만 보수적 코호트 하한(72s)으로 판정하면 미달
    m2 = compute_metrics(_line_run(70.0))
    assert m2["factory"]["lead_time"]["mean_s"] == 45.0
    assert m2["targets"]["leadtime_ok"] is False


def test_output_buffer_required_for_agv_departure():
    cfg = apply_overrides(rev_a(), {"stations.FA.output_buffer": None})
    with pytest.raises(ConfigError) as exc:
        resolve_config(cfg)
    assert any("FA.output_buffer" in e for e in exc.value.errors)


def test_write_run_dir_reuses_env_info(tmp_path, monkeypatch):
    from line_sim import run as v1_run
    from factory_sim.run import environment_info, write_run_dir
    env = environment_info()

    def no_git(*a, **k):
        raise AssertionError("git 호출 금지")

    monkeypatch.setattr(v1_run, "_git", no_git)
    r = run_simulation(single_line({"M": 1.0}, orders=explicit_orders(["A1"])), seed=1)
    write_run_dir(r, tmp_path / "x", env_info=env)
    assert (tmp_path / "x" / "metrics.json").exists()


def test_agv_loads_putter_items_at_same_stop():
    cfg = motor_config(40)
    cfg["stations"]["FA"]["output_buffer"] = {"capacity": 1}
    cfg["agv"]["count"] = 1
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    t = r.tables["agv_trips"]
    fa = t[t.pickup == "OUT_FA"]
    assert fa.n_items.max() > 1  # 출력 버퍼 1칸이지만 꺼낸 자리로 들어온 막힌 품목을 같은 정차에서 싣는다
    assert all(o.outcome != "incomplete" for o in r.orders)

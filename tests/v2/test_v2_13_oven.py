"""V13: 배치 오븐 — 가득 참/최대 대기 규칙, 배치 크기 상한, 고장 시 배치 전체 정지."""

from __future__ import annotations

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation


def _cfg(arrivals, *, failures=None):
    cfg = single_line({"OV": 0.0}, orders=explicit_orders(["A1"] * len(arrivals), arrivals), caps={"OV": 10})
    st = cfg["stations"]["OV"]
    st.pop("times_s")
    st["kind"] = "batch_oven"
    st["batch"] = {"max_size": 4, "process_s": 100.0, "max_wait_s": 50.0}
    if failures:
        st["failure_class"] = "o"
        cfg["failures"] = {"enabled": True, "distribution": "fixed", "mtbf_scale": 1.0, "classes": {"o": failures}}
    return cfg


def test_full_and_max_wait_rules():
    # 최대 대기 50초는 형성 시작(빈 챔버에 첫 품목 적재) 기준.
    # 0에 형성 시작(마감 50), 10·20·30 도착 → 30에 가득(4) → 30~130 가공, 130 하역.
    # 40 도착분은 가공 중 대기 → 130에 형성 시작(마감 180), 추가 도착 없음 → 180~280 단독.
    # 200 도착분은 가공 중 대기 → 280에 형성 시작(마감 330) → 330~430 단독.
    r = run_simulation(_cfg([0, 10, 20, 30, 40, 200]), seed=1, check_invariants=True, detail="full")
    b = r.tables["batches"]
    assert list(zip(b.start_s, b.end_s, b["size"])) == [(30.0, 130.0, 4), (180.0, 280.0, 1), (330.0, 430.0, 1)]
    assert sorted(o.terminal_s for o in r.orders) == [130.0] * 4 + [280.0, 430.0]
    assert b["size"].max() <= 4
    iv = r.intervals
    forming = iv[(iv.resource_id == "OV_1") & (iv.detail == "batch_forming")]
    assert list(zip(forming.start_s, forming.end_s)) == [(0.0, 30.0), (130.0, 180.0), (280.0, 330.0)]


def test_waiting_units_loaded_together_at_forming_start():
    # 0에 4개 → 즉시 가득, 0~100 가공. 10·20 도착분은 대기 → 100 하역 직후 함께 적재(형성 시작 100,
    # 마감 150). 140 도착분 추가 적재 → 3개로 150~250 가공.
    r = run_simulation(_cfg([0, 0, 0, 0, 10, 20, 140]), seed=1, check_invariants=True, detail="full")
    b = r.tables["batches"]
    assert list(zip(b.start_s, b.end_s, b["size"])) == [(0.0, 100.0, 4), (150.0, 250.0, 3)]


def test_failure_stops_whole_batch():
    r = run_simulation(_cfg([0, 0, 0, 0], failures={"mtbf_proc_s": 60.0, "mttr_s": 25.0, "repair_cv": 0.0}),
                       seed=1, detail="full")
    b = r.tables["batches"]
    assert list(zip(b.start_s, b.end_s, b["size"], b.down_s)) == [(0.0, 125.0, 4, 25.0)]
    assert [o.terminal_s for o in r.orders] == [125.0] * 4


def test_two_ovens_form_one_at_a_time():
    # 오븐 2대, 형성 토큰 1개. 0에 4개 → OV_1 즉시 가득, 0~100 가공(토큰은 0에 반환).
    # OV_2는 0에 토큰 확보·대기, 10·20·30 도착 → 10에 형성 시작(마감 60), 30에 3개, 60에 시작 60~160.
    # 70 도착분 → 토큰은 OV_1이 100 하역 후 확보(OV_2는 가공 중) → 100에 형성 시작(마감 150) → 150~250 단독.
    cfg = _cfg([0, 0, 0, 0, 10, 20, 30, 70])
    cfg["stations"]["OV"]["machines"] = 2
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    b = r.tables["batches"].sort_values("start_s")
    assert list(zip(b.machine_id, b.start_s, b.end_s, b["size"])) == [
        ("OV_1", 0.0, 100.0, 4), ("OV_2", 60.0, 160.0, 3), ("OV_1", 150.0, 250.0, 1)]
    # 형성 구간이 서로 겹치지 않는다
    iv = r.intervals
    f = iv[iv.detail == "batch_forming"].sort_values("start_s")
    assert all(a <= b_ for a, b_ in zip(f.end_s.iloc[:-1], f.start_s.iloc[1:]))

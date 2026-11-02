"""v2 설정: include 해석, override 규칙, 검증 오류, assumptions.csv와 Rev.A 일치."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from v2_helpers import REV_A_PATH, ROOT, rev_a
from factory_sim.config import ConfigError, apply_overrides, load_config, resolve_config, validate


def test_rev_a_valid_and_includes_resolved():
    cfg = load_config(REV_A_PATH)
    assert "includes" not in cfg
    assert len(cfg["layout"]["nodes"]) > 21 and len(cfg["operators"]["skills"]) == 8
    assert validate(cfg) == []
    assert sum(st["machines"] for st in cfg["stations"].values()) == 21


def test_overrides_open_prefixes_only():
    base = rev_a()
    out = apply_overrides(base, {"layout.nodes.R1_4": [10, 8], "stations.R1.machines": 4,
                                 "operators.skills.OP9": {"home": "R1_1", "skills": ["R1"]}, "time_scale.R1": 0.9})
    assert out["stations"]["R1"]["machines"] == 4 and "R1_4" in out["layout"]["nodes"]
    assert base["stations"]["R1"]["machines"] == 3  # 깊은 복사
    resolve_config(out)
    with pytest.raises(ConfigError):
        apply_overrides(base, {"stations.R1.speed": 2})
    with pytest.raises(ConfigError):
        apply_overrides(base, {"nope.x": 1})


def test_machine_added_without_layout_node_rejected():
    with pytest.raises(ConfigError) as exc:
        resolve_config(apply_overrides(rev_a(), {"stations.R1.machines": 4}))
    assert any("R1_4" in e for e in exc.value.errors)


@pytest.mark.parametrize("path,value", [
    ("stations.R1.machines", 0),
    ("stations.R1.times_s", {"A": -1, "B": 200, "C": 240}),
    ("products.groups.A.share", 0.6),
    ("products.groups.B.fail_rate", 1.5),
    ("release.kit_slots", 0),
    ("stations.RW.input_buffer", {"capacity": 5}),
    ("stations.EOL.input_buffer", {"capacity": 8, "reserved_recheck": 0}),
    ("stations.S3.dispatch", "LIFO"),
    ("stations.FA.operator", "sometimes"),
    ("agv.capacity", 0),
    ("demand.rate_per_h", 0),
    ("failures.mtbf_scale", 0),
    ("routing.ship_to", "MOON"),
    ("stations.R2.failure_class", "nope"),
])
def test_validate_blocks_bad_values(path, value):
    cfg = apply_overrides(rev_a(), {path: value})
    assert validate(cfg)
    with pytest.raises(ConfigError):
        resolve_config(cfg)


def test_operator_station_without_skill_rejected():
    cfg = copy.deepcopy(rev_a())
    for oc in cfg["operators"]["skills"].values():
        oc["skills"] = [s for s in oc["skills"] if s != "PK"]
    assert any("PK" in e for e in validate(cfg))


def test_assumptions_match_rev_a():
    df = pd.read_csv(ROOT / "data" / "v2" / "assumptions.csv", encoding="utf-8-sig", dtype=str)
    assert list(df.columns) == ["parameter", "value", "unit", "evidence_type", "source_or_reason", "limitation"]
    assert set(df.evidence_type) <= {"assumed", "design_choice", "derived"}
    v = dict(zip(df.parameter, df.value))
    cfg = resolve_config(REV_A_PATH)
    for sid, st in cfg["stations"].items():
        assert float(v[f"machines_{sid}"]) == st["machines"]
        for g, t in (st.get("times_s") or {}).items():
            assert float(v[f"{sid.lower()}_time_{g}"]) == t
        if st.get("setup_group_change_s"):
            assert float(v[f"{sid.lower()}_setup_group_change_s"]) == st["setup_group_change_s"]
    for g, gc in cfg["products"]["groups"].items():
        assert float(v[f"fail_rate_{g}"]) == gc["fail_rate"]
        assert float(v[f"share_{g}"]) == gc["share"]
    for c, cc in cfg["failures"]["classes"].items():
        assert float(v[f"mtbf_proc_{c}"]) == cc["mtbf_proc_s"]
        assert float(v[f"mttr_{c}"]) == cc["mttr_s"]
    assert float(v["kit_slots"]) == cfg["release"]["kit_slots"]
    assert float(v["demand_rate_per_h"]) == cfg["demand"]["rate_per_h"]
    assert float(v["agv_count"]) == cfg["agv"]["count"]
    assert float(v["operators_count"]) == len(cfg["operators"]["skills"])
    assert float(v["housing_reorder_point"]) == cfg["housing"]["reorder_point"]

"""v2 설정: 로드(include 해석), override, 검증, 접근자.

v1 ``line_sim.config``의 ConfigError·load_yaml·config_hash를 재사용한다. v1의
apply_overrides는 새 키를 ``time_scale`` 아래에서만 허용하므로 v2용으로 다시 만든다.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Mapping

from line_sim.config import ConfigError, config_hash, load_yaml  # noqa: F401 - re-export

# override가 새 키를 만들 수 있는 접두어 (예: layout.nodes.R1_4, operators.skills.OP9)
OPEN_PREFIXES = ("layout.nodes", "operators.skills", "time_scale")
OPERATOR_MODES = ("none", "setup_only", "full")
DISPATCH_RULES = ("FIFO", "EDD", "FAMILY")
STATION_KINDS = ("machine", "batch_oven", "vision_pick")   # vision_pick: PLAN_v5 S3 VP 셀
SPECIAL_NODES = ("RM", "HW", "KIT", "DOCK", "CHG")
PART_SOURCES = ("RM", "HW")
PART_SINKS = ("KIT", "SHIP")
SHIP_TARGETS = ("DOCK", "SHIP")
SLOT_RELEASE = ("FA_START", "TERMINAL")
WEEK_S = 288000.0  # 5일 × 2교대 × 8h 가동시간


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# ---------------------------------------------------------------------------- load / override
def load_config(path: str | Path) -> dict:
    """YAML을 읽고 ``includes: {dotted.path: 상대경로}``를 해석해 자기완결 설정을 반환한다."""
    path = Path(path)
    cfg = load_yaml(path)
    includes = cfg.pop("includes", None) or {}
    if not isinstance(includes, Mapping):
        raise ConfigError([f"{path}: includes는 mapping이어야 함"])
    for dotted, rel in includes.items():
        data = load_yaml(path.parent / str(rel))
        parts = str(dotted).split(".")
        node = cfg
        for key in parts[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigError([f"includes 대상이 mapping이 아님: {dotted}"])
        node[parts[-1]] = data
    return cfg


def apply_overrides(cfg: Mapping, overrides: Mapping[str, Any] | None) -> dict:
    """``{"dotted.path": value}`` override를 적용한 깊은 복사본을 반환한다.

    없는 키는 거부한다. 단, OPEN_PREFIXES 바로 아래의 새 키는 허용한다.
    """
    out = copy.deepcopy(dict(cfg))
    errors: list[str] = []
    for path, value in (overrides or {}).items():
        parts = str(path).split(".")
        node: Any = out
        ok = True
        for i, key in enumerate(parts[:-1]):
            if not isinstance(node, dict) or key not in node:
                errors.append(f"override 키 없음: {path} ({'.'.join(parts[:i + 1])})")
                ok = False
                break
            node = node[key]
        if not ok:
            continue
        last = parts[-1]
        if not isinstance(node, dict):
            errors.append(f"override 대상이 mapping이 아님: {path}")
            continue
        allow_new = ".".join(parts[:-1]) in OPEN_PREFIXES
        if last not in node and not allow_new:
            errors.append(f"override 키 없음: {path}")
            continue
        node[last] = copy.deepcopy(value)
    if errors:
        raise ConfigError(errors)
    return out


# ---------------------------------------------------------------------------- accessors
def groups(cfg: Mapping) -> list[str]:
    return list(cfg["products"]["groups"])


def sku_list(cfg: Mapping) -> list[str]:
    n = len(cfg["products"]["sku_factors"])
    return [f"{g}{i}" for g in groups(cfg) for i in range(1, n + 1)]


_SKU_RE = re.compile(r"^(.*?)(\d+)$")


def split_sku(sku: str) -> tuple[str, int]:
    """'A3' → ('A', 2) (군, 0부터 시작하는 SKU 계수 인덱스)."""
    m = _SKU_RE.match(str(sku))
    if not m:
        raise ConfigError([f"SKU 이름 형식 오류: {sku!r} (군 + 번호)"])
    return m.group(1), int(m.group(2)) - 1


def group_of(sku: str) -> str:
    return split_sku(sku)[0]


def station_ids(cfg: Mapping) -> list[str]:
    return list(cfg["stations"])


def machine_ids(cfg: Mapping) -> list[str]:
    return [f"{sid}_{k}" for sid, st in cfg["stations"].items() for k in range(1, st["machines"] + 1)]


def mean_time_s(cfg: Mapping, sid: str, sku: str) -> float:
    """군 평균 × SKU 계수 × time_scale (준비·오븐 고정시간에는 쓰지 않음)."""
    g, idx = split_sku(sku)
    st = cfg["stations"][sid]
    factor = float(cfg["products"]["sku_factors"][idx])
    return float(st["times_s"][g]) * factor * float((cfg.get("time_scale") or {}).get(sid, 1.0))


def window_s(cfg: Mapping) -> tuple[float, float]:
    w0 = float(cfg["run"]["warmup_s"])
    return w0, w0 + float(cfg["run"]["observation_s"])


def horizon_s(cfg: Mapping) -> float:
    return window_s(cfg)[1]


def assembly_stations(cfg: Mapping) -> list[str]:
    """조립 이후 공정 순서 [FA, EOL, PK] (+ RW는 inspect.rework)."""
    r = cfg["routing"]
    if not r.get("assembly"):
        return []
    insp = r.get("inspect") or {}
    out = [r["assembly"]]
    if insp:
        out += [insp["station"], insp["pass_to"]]
    return out


# ---------------------------------------------------------------------------- validation
def validate(cfg: Mapping) -> list[str]:
    """사람이 읽을 수 있는 오류 목록을 반환한다 (비어 있으면 유효)."""
    errors: list[str] = []
    if not isinstance(cfg, Mapping):
        return ["설정이 mapping이 아님"]

    run = cfg.get("run") or {}
    for key in ("warmup_s", "observation_s"):
        v = run.get(key)
        if not _is_number(v) or v < 0 or (key == "observation_s" and v <= 0):
            errors.append(f"run.{key}: {'양수' if key == 'observation_s' else '0 이상'}여야 함 ({v!r})")

    stoch = cfg.get("stochastic") or {}
    if not isinstance(stoch.get("enabled"), bool):
        errors.append("stochastic.enabled: true/false")
    if not _is_number(stoch.get("cv")) or stoch.get("cv") < 0:
        errors.append(f"stochastic.cv: 0 이상 ({stoch.get('cv')!r})")

    prod = cfg.get("products") or {}
    grp = prod.get("groups") or {}
    if not grp:
        errors.append("products.groups 비어 있음")
    shares = []
    for g, gc in grp.items():
        if not isinstance(gc, Mapping):
            errors.append(f"products.groups.{g}: mapping이 아님")
            continue
        for key in ("share", "fail_rate", "recover_prob"):
            v = gc.get(key)
            if not _is_number(v) or not (0 <= v <= 1):
                errors.append(f"products.groups.{g}.{key}: [0, 1] 범위 ({v!r})")
        if _is_number(gc.get("share")):
            shares.append(gc["share"])
    if shares and abs(sum(shares) - 1.0) > 1e-9:
        errors.append(f"products.groups 비율 합이 1이 아님 ({sum(shares)})")
    factors = prod.get("sku_factors")
    if not isinstance(factors, list) or not factors or not all(_is_number(f) and f > 0 for f in factors):
        errors.append(f"products.sku_factors: 양수 목록 ({factors!r})")

    stations = cfg.get("stations") or {}
    if not stations:
        errors.append("stations 비어 있음")
    routing = cfg.get("routing") or {}
    assembly = routing.get("assembly")
    insp = routing.get("inspect") or {}
    rework = insp.get("rework") if insp else None

    for sid, st in stations.items():
        if not isinstance(st, Mapping):
            errors.append(f"stations.{sid}: mapping이 아님")
            continue
        kind = st.get("kind", "machine")
        if kind not in STATION_KINDS:
            errors.append(f"{sid}.kind: {STATION_KINDS} 중 하나 ({kind!r})")
        m = st.get("machines")
        if not _is_int(m) or m < 1:
            errors.append(f"{sid}.machines: 1 이상의 정수 ({m!r})")
        if kind == "batch_oven":
            b = st.get("batch") or {}
            if not _is_int(b.get("max_size")) or b.get("max_size") < 1:
                errors.append(f"{sid}.batch.max_size: 1 이상의 정수")
            for key in ("process_s", "max_wait_s"):
                if not _is_number(b.get(key)) or b.get(key) < 0:
                    errors.append(f"{sid}.batch.{key}: 0 이상")
        elif kind == "vision_pick":
            vp = st.get("vision_pick") or {}
            if not vp.get("dist_file"):
                errors.append(f"{sid}.vision_pick.dist_file 없음")
            if not _is_int(vp.get("max_attempts")) or vp.get("max_attempts") < 1:
                errors.append(f"{sid}.vision_pick.max_attempts: 1 이상의 정수")
            for key in ("capture_s", "manual_s"):
                if not _is_number(vp.get(key)) or vp.get(key) < 0:
                    errors.append(f"{sid}.vision_pick.{key}: 0 이상")
            afr = vp.get("attempt_fail_rate")
            if afr is not None and (not _is_number(afr) or not 0 <= afr <= 1):
                errors.append(f"{sid}.vision_pick.attempt_fail_rate: 0~1")
        else:
            times = st.get("times_s") or {}
            for g in grp:
                t = times.get(g)
                if not _is_number(t) or t < 0:
                    errors.append(f"{sid}.times_s.{g}: 0 이상의 수 ({t!r})")
        s = st.get("setup_group_change_s", 0)
        if not _is_number(s) or s < 0:
            errors.append(f"{sid}.setup_group_change_s: 0 이상 ({s!r})")
        if st.get("operator", "none") not in OPERATOR_MODES:
            errors.append(f"{sid}.operator: {OPERATOR_MODES} 중 하나 ({st.get('operator')!r})")
        if st.get("dispatch", "FIFO") not in DISPATCH_RULES:
            errors.append(f"{sid}.dispatch: {DISPATCH_RULES} 중 하나 ({st.get('dispatch')!r})")
        fc = st.get("failure_class")
        if fc is not None and fc not in ((cfg.get("failures") or {}).get("classes") or {}):
            errors.append(f"{sid}.failure_class: failures.classes에 없음 ({fc!r})")
        if sid != assembly:
            ib = st.get("input_buffer")
            if not isinstance(ib, Mapping):
                errors.append(f"{sid}.input_buffer 없음")
            else:
                cap = ib.get("capacity")
                res = ib.get("reserved_recheck", 0)
                if cap is not None and (not _is_int(cap) or cap < 1):
                    errors.append(f"{sid}.input_buffer.capacity: 1 이상의 정수 또는 null ({cap!r})")
                if not _is_int(res) or res < 0 or (cap is not None and res >= cap):
                    errors.append(f"{sid}.input_buffer.reserved_recheck: 0 이상, capacity 미만 ({res!r})")
        ob = st.get("output_buffer")
        if ob is not None:
            cap = (ob or {}).get("capacity")
            if not _is_int(cap) or cap < 1:
                errors.append(f"{sid}.output_buffer.capacity: 1 이상의 정수 ({cap!r})")

    # 경로
    parts = routing.get("parts") or {}
    if not parts:
        errors.append("routing.parts 비어 있음")
    used: dict[str, str] = {}
    for p, pc in parts.items():
        if not isinstance(pc, Mapping):
            errors.append(f"routing.parts.{p}: mapping이 아님")
            continue
        if pc.get("from") not in PART_SOURCES:
            errors.append(f"routing.parts.{p}.from: {PART_SOURCES} 중 하나 ({pc.get('from')!r})")
        if pc.get("to") not in PART_SINKS:
            errors.append(f"routing.parts.{p}.to: {PART_SINKS} 중 하나 ({pc.get('to')!r})")
        path = pc.get("path")
        if not isinstance(path, list):
            errors.append(f"routing.parts.{p}.path: 목록이어야 함")
            continue
        for sid in path:
            if sid not in stations:
                errors.append(f"routing.parts.{p}.path: 없는 스테이션 {sid!r}")
            elif sid in used:
                errors.append(f"스테이션 {sid}가 여러 경로에 사용됨 ({used[sid]}, {p})")
            else:
                used[sid] = p
        if not path and pc.get("to") == "SHIP":
            errors.append(f"routing.parts.{p}: 빈 경로는 KIT로만 갈 수 있음")
    if assembly:
        if assembly not in stations:
            errors.append(f"routing.assembly: 없는 스테이션 {assembly!r}")
        if any((pc or {}).get("to") != "KIT" for pc in parts.values()):
            errors.append("assembly가 있으면 모든 부품의 to는 KIT")
        if not insp:
            errors.append("assembly가 있으면 routing.inspect 필요")
        else:
            for key in ("station", "rework", "pass_to"):
                if insp.get(key) not in stations:
                    errors.append(f"routing.inspect.{key}: 없는 스테이션 {insp.get(key)!r}")
            if insp.get("max_rework") not in (0, 1):
                errors.append("routing.inspect.max_rework: 0 또는 1")
        if routing.get("ship_to") not in SHIP_TARGETS:
            errors.append(f"routing.ship_to: {SHIP_TARGETS} 중 하나 ({routing.get('ship_to')!r})")
        for sid in [assembly, insp.get("station"), insp.get("rework"), insp.get("pass_to")]:
            if sid in used:
                errors.append(f"스테이션 {sid}가 부품 경로와 조립 이후 경로에 함께 사용됨")
    else:
        if len(parts) != 1 or any((pc or {}).get("to") != "SHIP" for pc in parts.values()):
            errors.append("assembly가 없으면 부품 1종, to: SHIP")
    if rework is not None and (stations.get(rework) or {}).get("input_buffer", {}).get("capacity") is not None:
        errors.append(f"{rework}.input_buffer.capacity: 재검사 교착 방지를 위해 null(무제한)이어야 함 (D2)")
    insp_sid = insp.get("station") if insp else None
    if insp_sid in stations and insp.get("max_rework") == 1:
        res = ((stations[insp_sid] or {}).get("input_buffer") or {}).get("reserved_recheck", 0)
        if not _is_int(res) or res < 1:
            errors.append(f"{insp_sid}.input_buffer.reserved_recheck: 재검사 예약 슬롯 1개 이상 필요")
    for sid in [assembly, insp_sid, rework]:
        if sid in stations and (stations[sid] or {}).get("kind", "machine") != "machine":
            errors.append(f"{sid}: 조립·검사·재작업은 일반 설비여야 함")

    rel = cfg.get("release") or {}
    k = rel.get("kit_slots")
    if not _is_int(k) or k < 1:
        errors.append(f"release.kit_slots: 1 이상의 정수 ({k!r})")
    if rel.get("slot_release_at") not in SLOT_RELEASE:
        errors.append(f"release.slot_release_at: {SLOT_RELEASE} 중 하나")
    if rel.get("slot_release_at") == "FA_START" and not assembly:
        errors.append("slot_release_at FA_START는 assembly가 있을 때만")

    dem = cfg.get("demand") or {}
    if dem.get("mode") not in ("poisson", "explicit"):
        errors.append(f"demand.mode: poisson | explicit ({dem.get('mode')!r})")
    if dem.get("mode") == "poisson" and (not _is_number(dem.get("rate_per_h")) or dem.get("rate_per_h") <= 0):
        errors.append(f"demand.rate_per_h: 양수 ({dem.get('rate_per_h')!r})")
    if not _is_number(dem.get("due_offset_s")) or dem.get("due_offset_s") < 0:
        errors.append("demand.due_offset_s: 0 이상")
    if dem.get("mode") == "explicit":
        ex = dem.get("explicit")
        if not isinstance(ex, list) or not ex:
            errors.append("demand.explicit: 주문 목록 필요")
        else:
            ids = [o.get("id") for o in ex if isinstance(o, Mapping)]
            if len(ids) != len(ex) or len(set(ids)) != len(ids):
                errors.append("demand.explicit: id가 고유해야 함")
            for o in ex:
                if not isinstance(o, Mapping):
                    continue
                try:
                    g, idx = split_sku(o.get("sku"))
                    if g not in grp or not (0 <= idx < len(factors or [])):
                        errors.append(f"demand.explicit: 없는 SKU {o.get('sku')!r}")
                except ConfigError as exc:
                    errors.extend(exc.errors)
                if not _is_number(o.get("arrival_s")) or o.get("arrival_s") < 0:
                    errors.append(f"demand.explicit: arrival_s 0 이상 ({o!r})")

    hs = cfg.get("housing") or {}
    if hs.get("enabled"):
        for key in ("reorder_point", "order_qty", "initial_on_hand"):
            if not _is_int(hs.get(key)) or hs.get(key) < (1 if key == "order_qty" else 0):
                errors.append(f"housing.{key}: 정수 ({hs.get(key)!r})")
        if not _is_number(hs.get("lead_time_s")) or hs.get("lead_time_s") < 0:
            errors.append("housing.lead_time_s: 0 이상")
        if not any((pc or {}).get("from") == "HW" for pc in parts.values()):
            errors.append("housing.enabled인데 HW에서 오는 부품이 없음")

    fl = cfg.get("failures") or {}
    if not isinstance(fl.get("enabled"), bool):
        errors.append("failures.enabled: true/false")
    if fl.get("distribution") not in ("stochastic", "fixed"):
        errors.append("failures.distribution: stochastic | fixed")
    if not _is_number(fl.get("mtbf_scale")) or fl.get("mtbf_scale") <= 0:
        errors.append("failures.mtbf_scale: 양수")
    for c, cc in (fl.get("classes") or {}).items():
        for key in ("mtbf_proc_s", "mttr_s", "repair_cv"):
            v = (cc or {}).get(key)
            if not _is_number(v) or v < 0 or (key == "mtbf_proc_s" and v <= 0):
                errors.append(f"failures.classes.{c}.{key}: 범위 오류 ({v!r})")

    ops = cfg.get("operators") or {}
    if ops.get("mode") not in ("pool", "unlimited"):
        errors.append("operators.mode: pool | unlimited")
    layout_nodes = (cfg.get("layout") or {}).get("nodes") or {}
    if ops.get("mode") == "pool":
        if not _is_number(ops.get("walk_speed_mps")) or ops.get("walk_speed_mps") <= 0:
            errors.append("operators.walk_speed_mps: 양수")
        skills = ops.get("skills") or {}
        covered: set[str] = set()
        for op, oc in skills.items():
            if not isinstance(oc, Mapping) or oc.get("home") not in layout_nodes:
                errors.append(f"operators.skills.{op}.home: 레이아웃 노드여야 함")
                continue
            for sid in oc.get("skills") or []:
                if sid not in stations:
                    errors.append(f"operators.skills.{op}: 없는 공정 {sid!r}")
                covered.add(sid)
        for sid, st in stations.items():
            if isinstance(st, Mapping) and st.get("operator", "none") != "none" and sid not in covered:
                errors.append(f"{sid}: 작업자가 필요한데 숙련자가 없음")

    agv = cfg.get("agv") or {}
    if agv.get("mode") not in ("fleet", "instant"):
        errors.append("agv.mode: fleet | instant")
    if agv.get("mode") == "fleet":
        if not _is_int(agv.get("count")) or agv.get("count") < 1:
            errors.append("agv.count: 1 이상의 정수")
        if not _is_int(agv.get("capacity")) or agv.get("capacity") < 1:
            errors.append("agv.capacity: 1 이상의 정수")
        for key in ("speed_mps",):
            if not _is_number(agv.get(key)) or agv.get(key) <= 0:
                errors.append(f"agv.{key}: 양수")
        for key in ("load_s", "unload_s"):
            if not _is_number(agv.get(key)) or agv.get(key) < 0:
                errors.append(f"agv.{key}: 0 이상")
        ch = agv.get("charge") or {}
        for key in ("after_busy_s", "duration_s"):
            if not _is_number(ch.get(key)) or ch.get(key) < 0:
                errors.append(f"agv.charge.{key}: 0 이상")
        if agv.get("home_node") not in layout_nodes or ch.get("node") not in layout_nodes:
            errors.append("agv.home_node, agv.charge.node: 레이아웃 노드여야 함")
        # AGV 운반 지점 노드
        needed: set[str] = set()
        if assembly and routing.get("ship_to") == "DOCK":
            needed.add("DOCK")
        for p, pc in parts.items():
            if not isinstance(pc, Mapping):
                continue
            needed.add(str(pc.get("from")))
            if pc.get("to") == "KIT":
                needed.add("KIT")
            if pc.get("path"):
                needed.add(f"{pc['path'][0]}_IN")
                if pc.get("to") == "KIT":
                    needed.add(f"{pc['path'][-1]}_OUT")
        if assembly and insp:
            needed |= {f"{assembly}_OUT", f"{insp['station']}_IN", f"{insp['station']}_OUT",
                       f"{insp['rework']}_IN", f"{insp['rework']}_OUT", f"{insp['pass_to']}_IN"}
            if routing.get("ship_to") == "DOCK":
                needed.add(f"{insp['pass_to']}_OUT")
        for n in sorted(needed):
            if n not in layout_nodes:
                errors.append(f"layout.nodes.{n}: AGV 운반 지점 노드 없음")
        # AGV가 출발하는 스테이션은 출력 버퍼 용량을 명시해야 한다(조용한 기본값 없음).
        for n in sorted(needed):
            sid = n[:-4] if n.endswith("_OUT") else None
            if sid in stations and not isinstance((stations[sid] or {}).get("output_buffer"), Mapping):
                errors.append(f"{sid}.output_buffer.capacity: AGV 출발 스테이션은 필수")

    if ops.get("mode") == "pool" or agv.get("mode") == "fleet":
        for mid in machine_ids_safe(stations):
            if mid not in layout_nodes:
                errors.append(f"layout.nodes.{mid}: 설비 노드 없음")
        for n, xy in layout_nodes.items():
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2 and all(_is_number(v) for v in xy)):
                errors.append(f"layout.nodes.{n}: [x, y] 좌표여야 함")
        d = (cfg.get("layout") or {}).get("detour_factor")
        if not _is_number(d) or d <= 0:
            errors.append("layout.detour_factor: 양수")

    disp = cfg.get("dispatch") or {}
    if not _is_number(disp.get("t_max_s")) or disp.get("t_max_s") < 0:
        errors.append("dispatch.t_max_s: 0 이상")

    scale = cfg.get("time_scale") or {}
    if not isinstance(scale, Mapping):
        errors.append("time_scale: mapping")
    else:
        for sid, v in scale.items():
            if sid not in stations:
                errors.append(f"time_scale.{sid}: 없는 스테이션")
            if not _is_number(v) or v <= 0:
                errors.append(f"time_scale.{sid}: 양수 ({v!r})")
    return errors


def machine_ids_safe(stations: Mapping) -> list[str]:
    out = []
    for sid, st in stations.items():
        m = (st or {}).get("machines") if isinstance(st, Mapping) else None
        if _is_int(m):
            out += [f"{sid}_{k}" for k in range(1, m + 1)]
    return out


def resolve_config(path_or_dict: str | Path | Mapping, overrides: Mapping[str, Any] | None = None) -> dict:
    """로드(경로면) → override → 검증. 실패 시 ConfigError."""
    base = load_config(path_or_dict) if isinstance(path_or_dict, (str, Path)) else path_or_dict
    cfg = apply_overrides(base, overrides)
    if cfg.get("time_scale") is None:
        cfg["time_scale"] = {}
    try:
        errors = validate(cfg)
    except (AttributeError, TypeError, KeyError) as exc:  # 구조가 깨진 설정
        raise ConfigError([f"설정 구조 오류: {exc!r}"]) from exc
    if errors:
        raise ConfigError(errors)
    return cfg

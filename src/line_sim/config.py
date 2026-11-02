"""Scenario configuration: loading, overrides, validation and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

TERMINAL_TARGETS = ("SHIP", "SCRAP")
ROUTE_TYPES = ("next", "inspect", "rework")
SEQUENCE_POLICIES = ("random_permutation", "grouped")
# Top-level sections under which an override may create a new key.
OPEN_SECTIONS = ("time_scale",)


class ConfigError(ValueError):
    """Raised when a configuration fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("설정 오류:\n- " + "\n- ".join(self.errors))


def load_yaml(path: str | Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError([f"{path}: 파일을 읽을 수 없음 ({exc.strerror or exc})"]) from exc
    except yaml.YAMLError as exc:
        raise ConfigError([f"{path}: YAML 형식 오류 ({exc})"]) from exc
    if not isinstance(data, dict):
        raise ConfigError([f"{path}: YAML 최상위가 mapping이 아님"])
    return data


def apply_overrides(cfg: Mapping, overrides: Mapping[str, Any] | None) -> dict:
    """Return a deep copy of ``cfg`` with ``{"dotted.path": value}`` overrides applied.

    Unknown keys are rejected, except new keys directly under OPEN_SECTIONS
    (e.g. ``time_scale.ASM``).
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
        allow_new = len(parts) == 2 and parts[0] in OPEN_SECTIONS
        if last not in node and not allow_new:
            errors.append(f"override 키 없음: {path}")
            continue
        node[last] = copy.deepcopy(value)
    if errors:
        raise ConfigError(errors)
    return out


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def product_types(cfg: Mapping) -> list[str]:
    """Product types in canonical order (order of ``orders.counts``)."""
    return list((cfg.get("orders") or {}).get("counts") or {})


def validate(cfg: Mapping) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errors: list[str] = []
    if not isinstance(cfg, Mapping):
        return ["설정이 mapping이 아님"]

    orders = cfg.get("orders") or {}
    counts = orders.get("counts") or {}
    products = list(counts)
    if not counts:
        errors.append("orders.counts 비어 있음")
    total = 0
    for p, n in counts.items():
        if not _is_int(n) or n < 0:
            errors.append(f"orders.counts.{p}: 0 이상의 정수여야 함 ({n!r})")
        else:
            total += n
    if counts and total <= 0:
        errors.append("주문 합계가 0 — 빈 주문")
    for key in ("order_ready_at_s", "due_s"):
        v = orders.get(key)
        if not _is_number(v) or v < 0:
            errors.append(f"orders.{key}: 0 이상의 수여야 함 ({v!r})")

    release = cfg.get("release") or {}
    mi = release.get("min_interval_s")
    if not _is_number(mi) or mi < 0:
        errors.append(f"release.min_interval_s: 0 이상이어야 함 ({mi!r})")
    wl = release.get("wip_limit")
    if wl is not None and (not _is_int(wl) or wl < 1):
        errors.append(f"release.wip_limit: 1 이상의 정수 또는 null ({wl!r})")
    seq = release.get("sequence") or {}
    policy = seq.get("policy")
    if policy not in SEQUENCE_POLICIES:
        errors.append(f"release.sequence.policy: {SEQUENCE_POLICIES} 중 하나 ({policy!r})")
    if policy == "grouped":
        order = seq.get("order")
        if not isinstance(order, list) or sorted(order) != sorted(products):
            errors.append(f"release.sequence.order: 모든 품종을 한 번씩 포함해야 함 ({order!r})")

    run = cfg.get("run") or {}
    win = run.get("observation_window_s")
    if (not isinstance(win, (list, tuple)) or len(win) != 2
            or not all(_is_number(x) for x in win) or not (0 <= win[0] < win[1])):
        errors.append(f"run.observation_window_s: [w0, w1], 0 <= w0 < w1 ({win!r})")
    tl = run.get("time_limit_s")
    if not _is_number(tl) or tl <= 0:
        errors.append(f"run.time_limit_s: 양수여야 함 ({tl!r})")
    elif (isinstance(win, (list, tuple)) and len(win) == 2 and _is_number(win[1]) and tl < win[1]):
        errors.append(f"run.time_limit_s({tl})가 관측창 끝({win[1]})보다 작음")

    stoch = cfg.get("stochastic") or {}
    if not isinstance(stoch.get("enabled"), bool):
        errors.append("stochastic.enabled: true/false")
    if stoch.get("distribution", "lognormal") != "lognormal":
        errors.append("stochastic.distribution: lognormal만 지원")
    cv = stoch.get("cv")
    if not _is_number(cv) or cv < 0:
        errors.append(f"stochastic.cv: 0 이상이어야 함 ({cv!r})")

    prod_cfg = cfg.get("products") or {}
    for p in products:
        pc = prod_cfg.get(p)
        if not isinstance(pc, Mapping):
            errors.append(f"products.{p}: 품질 설정 없음")
            continue
        for key in ("fail_rate", "recover_prob"):
            v = pc.get(key)
            if not _is_number(v) or not (0 <= v <= 1):
                errors.append(f"products.{p}.{key}: [0, 1] 범위여야 함 ({v!r})")

    line = cfg.get("line") or {}
    stations = line.get("stations") or {}
    if not stations:
        errors.append("line.stations 비어 있음")
    if line.get("first_station") not in stations:
        errors.append(f"line.first_station: 존재하는 스테이션이어야 함 ({line.get('first_station')!r})")
    for sid, st in stations.items():
        if not isinstance(st, Mapping):
            errors.append(f"line.stations.{sid}: mapping이 아님")
            continue
        m = st.get("machines")
        if not _is_int(m) or m < 1:
            errors.append(f"{sid}.machines: 1 이상의 정수여야 함 ({m!r})")
        buf = st.get("input_buffer") or {}
        if not buf.get("id"):
            errors.append(f"{sid}.input_buffer.id 없음")
        cap = buf.get("capacity")
        if cap is not None and (not _is_int(cap) or cap < 1):
            errors.append(f"{sid}.input_buffer.capacity: 1 이상의 정수 또는 null ({cap!r})")
        times = st.get("times_s") or {}
        for p in products:
            t = times.get(p)
            if not _is_number(t) or t < 0:
                errors.append(f"{sid}.times_s.{p}: 0 이상의 수여야 함 ({t!r})")
        matrix = st.get("setup_matrix_s")
        if matrix is not None:
            if not isinstance(matrix, Mapping) or set(matrix) != set(products):
                errors.append(f"{sid}.setup_matrix_s: 모든 품종 행이 필요함")
            else:
                for a, row in matrix.items():
                    if not isinstance(row, Mapping) or set(row) != set(products):
                        errors.append(f"{sid}.setup_matrix_s.{a}: 정사각 행렬이어야 함")
                        continue
                    for b, v in row.items():
                        if not _is_number(v) or v < 0:
                            errors.append(f"{sid}.setup_matrix_s.{a}.{b}: 0 이상 ({v!r})")
        route = st.get("route") or {}
        rtype = route.get("type", "next")
        if rtype not in ROUTE_TYPES:
            errors.append(f"{sid}.route.type: {ROUTE_TYPES} 중 하나 ({rtype!r})")
        targets = [route.get("next")] if rtype == "next" else [route.get("on_pass"), route.get("on_fail")]
        for tgt in targets:
            if tgt not in stations and tgt not in TERMINAL_TARGETS:
                errors.append(f"{sid}.route 대상이 존재하지 않음 ({tgt!r})")

    errors.extend(_route_errors(line.get("first_station"), stations))

    scale = cfg.get("time_scale") or {}
    if not isinstance(scale, Mapping):
        errors.append("time_scale: mapping이어야 함")
    else:
        for sid, v in scale.items():
            if sid not in stations:
                errors.append(f"time_scale.{sid}: 존재하지 않는 스테이션")
            if not _is_number(v) or v <= 0:
                errors.append(f"time_scale.{sid}: 양수여야 함 ({v!r})")
    return errors


def _route_targets(st: Mapping) -> list[Any]:
    route = st.get("route") or {}
    if route.get("type", "next") == "next":
        return [route.get("next")]
    return [route.get("on_pass"), route.get("on_fail")]


def _route_errors(first: Any, stations: Mapping) -> list[str]:
    """Reject cycles reachable from the first station and more than one inspect/rework station."""
    errors: list[str] = []
    for rtype in ("inspect", "rework"):
        ids = [sid for sid, st in stations.items()
               if isinstance(st, Mapping) and (st.get("route") or {}).get("type") == rtype]
        if len(ids) > 1:
            errors.append(f"route.type={rtype} 스테이션은 최대 1개 ({ids})")
    if first not in stations:
        return errors
    color: dict[str, int] = {}  # 1 = on stack, 2 = done

    def dfs(sid: str) -> bool:
        color[sid] = 1
        st = stations[sid]
        for tgt in (_route_targets(st) if isinstance(st, Mapping) else []):
            if tgt not in stations:
                continue
            if color.get(tgt) == 1:
                errors.append(f"route 순환: {sid} → {tgt}")
                return True
            if tgt not in color and dfs(tgt):
                return True
        color[sid] = 2
        return False

    dfs(first)
    return errors


def resolve_config(path_or_dict: str | Path | Mapping, overrides: Mapping[str, Any] | None = None) -> dict:
    """Load (if path), apply overrides and validate. Raises ConfigError."""
    base = load_yaml(path_or_dict) if isinstance(path_or_dict, (str, Path)) else path_or_dict
    cfg = apply_overrides(base, overrides)
    if "time_scale" not in cfg or cfg["time_scale"] is None:
        cfg["time_scale"] = {}
    try:
        errors = validate(cfg)
    except (AttributeError, TypeError, KeyError) as exc:  # malformed structure (e.g. list instead of mapping)
        raise ConfigError([f"설정 구조 오류: {exc!r}"]) from exc
    if errors:
        raise ConfigError(errors)
    return cfg


def config_hash(cfg: Mapping) -> str:
    text = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nominal_capacity(cfg: Mapping) -> dict[str, float | None]:
    """Long-run reference capacity (units/hour) ignoring setup, variability, buffers and defects.

    Mean time is weighted by the order mix. Rework-type stations are skipped
    because only failed units visit them. ``release`` = 3600 / min_interval.
    """
    counts = cfg["orders"]["counts"]
    total = sum(counts.values())
    scale = cfg.get("time_scale") or {}
    out: dict[str, float | None] = {}
    for sid, st in cfg["line"]["stations"].items():
        if (st.get("route") or {}).get("type") == "rework":
            continue
        mean_t = sum(counts[p] / total * st["times_s"][p] for p in counts) * scale.get(sid, 1.0)
        out[sid] = None if mean_t <= 0 else st["machines"] * 3600.0 / mean_t
    mi = cfg["release"]["min_interval_s"]
    out["release"] = None if mi <= 0 else 3600.0 / mi
    return out

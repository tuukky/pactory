"""모터 공장 v2 Streamlit 화면 (3개 탭). 저장된 결과만 읽는다 — 시뮬레이션 실행 로직은 없다.

실행: python -m streamlit run app/dashboard_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import factory_sim  # noqa: F401
except ImportError:  # 설치 없이 실행할 때의 fallback
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factory_sim.config import resolve_config
from factory_sim.static_capacity import compute as static_compute

CONFIGS = ROOT / "configs" / "v2"
RESULTS = ROOT / "results" / "v2"
ENC = "utf-8-sig"
BANNER = "가상 공장 / 합성 가정 / 실제 공장 성능 검증 아님"
NO_CLEAR_DIFF = "이번 실험으로 명확한 차이를 확인하지 못함"
REVS = {"A": "rev_A.yaml", "B": "rev_B.yaml", "C": "rev_C.yaml"}

# 손실 분해 7범주 (PLAN_v2 §6)
STATES = ["PROCESSING", "SETUP", "STARVED", "BLOCKED_DOWNSTREAM", "BLOCKED_TRANSPORT", "WAIT_OPERATOR", "DOWN"]
STATE_LABELS = {"PROCESSING": "가공", "SETUP": "준비", "STARVED": "굶주림", "BLOCKED_DOWNSTREAM": "막힘(하류)",
                "BLOCKED_TRANSPORT": "막힘(운반)", "WAIT_OPERATOR": "작업자 대기", "DOWN": "고장"}
STATE_COLORS = {"PROCESSING": "#2a9d8f", "SETUP": "#e9c46a", "STARVED": "#adb5bd", "BLOCKED_DOWNSTREAM": "#e76f51",
                "BLOCKED_TRANSPORT": "#f4a261", "WAIT_OPERATOR": "#8338ec", "DOWN": "#d62828"}
OP_STATES = ["IDLE", "WALK", "SETUP", "PROCESS", "WAIT_REPAIR"]
AGV_STATES = ["IDLE", "TRAVEL_EMPTY", "LOAD", "TRAVEL_LOADED", "UNLOAD", "BLOCKED", "TO_CHARGE", "CHARGING"]
METRIC_LABELS = {"factory.demand_fulfil_ratio": "수요 충족률", "factory.otd_rate": "납기 충족률",
                 "factory.lead_time_cohort_mean_s": "리드타임 하한(s)"}

st.set_page_config(page_title="모터 공장 v2 공정설계", layout="wide")


# ----------------------------------------------------------------------------- loaders
@st.cache_data(show_spinner=False)
def load_rev(rev: str) -> dict:
    return resolve_config(CONFIGS / REVS[rev])


@st.cache_data(show_spinner=False)
def load_static(rev: str) -> dict:
    return static_compute(load_rev(rev))


def list_full_runs() -> dict[str, Path]:
    """metrics.json과 resource_intervals.csv가 있는 full 상세 run 목록."""
    runs: dict[str, Path] = {}
    for base in (RESULTS / "representative", RESULTS / "demo"):
        if not base.exists():
            continue
        for p in sorted(base.glob("**/metrics.json")):
            d = p.parent
            if (d / "resource_intervals.csv").exists():
                runs[str(d.relative_to(RESULTS)).replace("\\", "/")] = d
    return runs


@st.cache_data(show_spinner=False)
def load_metrics(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str, usecols: tuple | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding=ENC, usecols=list(usecols) if usecols else None, low_memory=False)


def latest_experiment(prefix: str) -> Path | None:
    if not RESULTS.exists():
        return None
    dirs = sorted(d for d in RESULTS.iterdir() if d.is_dir() and d.name.startswith(prefix))
    return dirs[-1] if dirs else None


def no_results(msg: str) -> None:
    st.info(f"{msg}\n\n결과가 없습니다. 먼저 실험을 실행하세요 (README의 'v2 모터 공장' 절 참고):\n\n"
            "`python -m factory_sim.run --config configs/v2/rev_C.yaml --seed 100 "
            "--output results/v2/representative/rev_C --detail full`")


def node_xy(layout: dict, name: str) -> tuple[float, float] | None:
    """버퍼 ID(OUT_R4 등)를 레이아웃 노드로 바꾼다."""
    nodes = layout["nodes"]
    if name in nodes:
        return tuple(nodes[name])
    if name.startswith("OUT_"):
        sid = name[4:]
        for cand in (sid, f"{sid}_OUT"):
            if cand in nodes:
                return tuple(nodes[cand])
    return None


# ----------------------------------------------------------------------------- figures
# AGV 구간(설명용 직선)
LANES = [("RM", "R1_IN"), ("RM", "S1_IN"), ("HW", "KIT"), ("R4_OUT", "KIT"), ("S5_OUT", "KIT"),
         ("FA_OUT", "EOL_IN"), ("EOL_OUT", "PK_IN"), ("EOL_OUT", "RW_IN"), ("RW_OUT", "EOL_IN"),
         ("PK_OUT", "DOCK")]


def layout_figure(cfg: dict, machine_state: dict[str, str] | None = None,
                  agv_xy: dict[str, tuple] | None = None, op_xy: dict[str, tuple] | None = None) -> go.Figure:
    nodes = cfg["layout"]["nodes"]
    fig = go.Figure()
    for a, b in LANES:
        if a in nodes and b in nodes:
            fig.add_trace(go.Scatter(x=[nodes[a][0], nodes[b][0]], y=[nodes[a][1], nodes[b][1]], mode="lines",
                                     line=dict(color="#ced4da", dash="dot"), hoverinfo="skip", showlegend=False))
    mach = {k: v for k, v in nodes.items() if "_" in k and k.rsplit("_", 1)[1].isdigit()}
    other = {k: v for k, v in nodes.items() if k not in mach}
    colors = [STATE_COLORS.get((machine_state or {}).get(k, ""), "#457b9d") for k in mach]
    hover = [f"{k}: {STATE_LABELS.get((machine_state or {}).get(k, ''), '')}" for k in mach]
    fig.add_trace(go.Scatter(x=[v[0] for v in mach.values()], y=[v[1] for v in mach.values()], mode="markers+text",
                             text=list(mach), textposition="top center", hovertext=hover, hoverinfo="text",
                             marker=dict(size=16, symbol="square", color=colors), name="설비"))
    fig.add_trace(go.Scatter(x=[v[0] for v in other.values()], y=[v[1] for v in other.values()], mode="markers+text",
                             text=list(other), textposition="bottom center", marker=dict(size=9, color="#6c757d"),
                             name="버퍼·창고·도크"))
    if agv_xy:
        fig.add_trace(go.Scatter(x=[p[0] for p in agv_xy.values()], y=[p[1] for p in agv_xy.values()],
                                 mode="markers+text", text=list(agv_xy), textposition="middle right",
                                 marker=dict(size=13, symbol="diamond", color="#f77f00"), name="AGV"))
    if op_xy:
        fig.add_trace(go.Scatter(x=[p[0] + 1.2 for p in op_xy.values()], y=[p[1] - 1.2 for p in op_xy.values()],
                                 mode="markers+text", text=list(op_xy), textposition="bottom right",
                                 marker=dict(size=10, symbol="circle", color="#3a0ca3"), name="작업자"))
    fig.update_layout(height=520, xaxis_title="x (m, 설명용)", yaxis_title="y (m)",
                      yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    return fig


def process_graph_figure(cfg: dict) -> go.Figure:
    parts = cfg["routing"]["parts"]
    pos, edges = {}, []
    for row, (part, pc) in enumerate(parts.items()):
        chain = [pc["from"]] + list(pc["path"]) + ["KIT"]
        for i, n in enumerate(chain):
            if n != "KIT":
                pos.setdefault(n, (float(i), -float(row)))
        edges += list(zip(chain[:-1], chain[1:]))
    pos["KIT"] = (6.0, -1.0)
    asm, insp = cfg["routing"]["assembly"], cfg["routing"]["inspect"]
    pos[asm], pos[insp["station"]], pos[insp["rework"]] = (7.0, -1.0), (8.0, -1.0), (8.0, -2.3)
    pos[insp["pass_to"]], pos["DOCK"] = (9.0, -1.0), (10.0, -1.0)
    edges += [("KIT", asm), (asm, insp["station"]), (insp["station"], insp["pass_to"]),
              (insp["station"], insp["rework"]), (insp["rework"], insp["station"]), (insp["pass_to"], "DOCK")]
    fig = go.Figure()
    for a, b in edges:
        fig.add_annotation(x=pos[b][0], y=pos[b][1], ax=pos[a][0], ay=pos[a][1], xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#6c757d")
    st_cfg = cfg["stations"]
    labels = [f"{n}<br>×{st_cfg[n]['machines']}" if n in st_cfg else n for n in pos]
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode="markers+text",
                             text=labels, textposition="top center", marker=dict(size=22, color="#a8dadc"),
                             hoverinfo="text", showlegend=False))
    fig.update_layout(height=330, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def stacked_states(table: dict[str, dict], states: list[str], labels: dict | None = None,
                   colors: dict | None = None, height: int = 420) -> go.Figure:
    ids = list(table)
    fig = go.Figure()
    for s in states:
        fig.add_trace(go.Bar(name=(labels or {}).get(s, s), x=ids, y=[100 * (table[i].get(s) or 0.0) for i in ids],
                             marker_color=(colors or {}).get(s)))
    fig.update_layout(barmode="stack", height=height, yaxis_title="관측시간 비율 (%)",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def waterfall_figure(m: dict, cfg: dict, sid: str) -> go.Figure:
    """선택 공정의 이론 능력(대/h)에서 손실 상태 비율만큼 빼서 가공 상당 능력과 실제 출하를 비교한다."""
    srow = static_compute(cfg)["stations"].set_index("station")
    cap = float(srow.loc[sid, "cap_pure_per_h"]) if sid in srow.index else float("nan")
    shares = m["stations"][sid]
    losses = [s for s in STATES if s != "PROCESSING"]
    x = ["이론 능력"] + [STATE_LABELS[s] for s in losses] + ["가공 상당", "실제 양품 출하"]
    y = [cap] + [-cap * (shares.get(s) or 0.0) for s in losses] + [0, m["factory"]["good_rate_per_h"]]
    measure = ["absolute"] + ["relative"] * len(losses) + ["total", "absolute"]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=measure, text=[f"{v:,.1f}" for v in y],
                                 decreasing=dict(marker=dict(color="#e76f51")),
                                 totals=dict(marker=dict(color="#2a9d8f"))))
    fig.update_layout(height=380, yaxis_title="대/h", margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ----------------------------------------------------------------------------- playback
@st.cache_data(show_spinner=False)
def machine_state_at(run_dir: str, t: float) -> dict[str, str]:
    ri = load_csv(str(Path(run_dir) / "resource_intervals.csv"),
                  ("resource_type", "resource_id", "state", "start_s", "end_s"))
    mi = ri[(ri.resource_type == "machine") & (ri.start_s <= t) & (ri.end_s > t)]
    return dict(zip(mi.resource_id, mi.state))


@st.cache_data(show_spinner=False)
def agv_tables(run_dir: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """AGV별 (상태 구간, 운반 기록). 재생 프레임마다 전체 표를 거르지 않도록 한 번만 나눈다."""
    trips = load_csv(str(Path(run_dir) / "agv_trips.csv"))
    ri = load_csv(str(Path(run_dir) / "resource_intervals.csv"),
                  ("resource_type", "resource_id", "state", "start_s", "end_s"))
    ri = ri[ri.resource_type == "agv"]
    return {agv: (ri[ri.resource_id == agv], trips[trips.agv_id == agv]) for agv in sorted(ri.resource_id.unique())}


def agv_positions_at(run_dir: str, cfg: dict, t: float) -> dict[str, tuple]:
    lay = cfg["layout"]
    chg = node_xy(lay, cfg["agv"].get("home_node", "CHG")) or (0.0, 0.0)
    out: dict[str, tuple] = {}
    for agv, (ri, trips) in agv_tables(run_dir).items():
        cur = ri[(ri.start_s <= t) & (ri.end_s > t)]
        state = cur.state.iloc[0] if len(cur) else "IDLE"
        tr = trips[trips.t_assign <= t]
        if state in ("CHARGING", "TO_CHARGE") or tr.empty:
            out[agv] = chg
            continue
        r = tr.iloc[-1]
        pick, dest = node_xy(lay, r.pickup) or chg, node_xy(lay, r.dest) or chg
        if t < r.t_arrive_pick:
            prev = node_xy(lay, tr.iloc[-2].dest) if len(tr) > 1 else chg
            a, b, t0, t1 = prev or chg, pick, r.t_assign, r.t_arrive_pick
        elif t < r.t_depart:
            out[agv] = pick
            continue
        elif t < r.t_arrive_dest:
            a, b, t0, t1 = pick, dest, r.t_depart, r.t_arrive_dest
        else:
            out[agv] = dest
            continue
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out[agv] = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
    return out


def operator_positions_at(run_dir: str, cfg: dict, t: float) -> dict[str, tuple]:
    """작업자 위치 = 마지막 배정 설비 노드(도보 중 보간 없음), 배정 전에는 home."""
    ev = load_csv(str(Path(run_dir) / "events.csv"), ("time_s", "event_type", "resource_id", "detail"))
    lay = cfg["layout"]
    out = {op: node_xy(lay, oc["home"]) for op, oc in cfg["operators"]["skills"].items()}
    if not ev.empty:
        a = ev[(ev.event_type == "OP_ASSIGN") & (ev.time_s <= t)]
        for op, node in a.groupby("resource_id")["detail"].last().items():
            xy = node_xy(lay, str(node))
            if xy:
                out[op] = xy
    return {k: v for k, v in out.items() if v}


@st.cache_data(show_spinner=False)
def occupancy_series(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """키트 점유(투입~FA 시작)와 버퍼 대기 수(BUFFER_ENTER/EXIT의 buffer_count) 계단 시계열."""
    orders = load_csv(str(Path(run_dir) / "orders.csv"))
    kit = pd.DataFrame()
    if not orders.empty:
        rel = orders.released_s.dropna()
        fa = orders.fa_start_s.dropna()
        steps = pd.concat([pd.Series(1, index=rel.values), pd.Series(-1, index=fa.values)]).sort_index()
        kit = pd.DataFrame({"time_h": steps.index / 3600.0, "kit_occupied": steps.cumsum().values})
    ev = load_csv(str(Path(run_dir) / "events.csv"), ("time_s", "event_type", "buffer_id", "buffer_count"))
    buf = pd.DataFrame()
    if not ev.empty:
        buf = ev[ev.event_type.isin(["BUFFER_ENTER", "BUFFER_EXIT"])][["time_s", "buffer_id", "buffer_count"]]
        buf = buf.assign(time_h=buf.time_s / 3600.0)
    return kit, buf


def _step_lookup(times: np.ndarray, values: np.ndarray, ts: np.ndarray, default):
    """계단 시계열(times 오름차순)에서 각 ts 시각의 값."""
    idx = np.searchsorted(times, ts, side="right") - 1
    return [values[i] if i >= 0 else default for i in idx]


GROUP_COLORS = {"A": "#0077b6", "B": "#7209b7", "C": "#ff006e"}
OP_COLORS = {"PROCESS": "#3a0ca3", "SETUP": "#e9c46a", "WALK": "#06d6a0", "IDLE": "#adb5bd", "WAIT_REPAIR": "#d62828"}
OP_LABELS = {"PROCESS": "가공", "SETUP": "준비", "WALK": "이동", "IDLE": "대기", "WAIT_REPAIR": "수리 대기"}
BUF_SLOTS, BUF_COLS, BUF_GAP = 24, 6, 0.75  # 버퍼당 점 최대 개수·열 수·간격(m). 넘으면 숫자로만 표시


def buffer_xy(lay: dict, bid: str) -> tuple[float, float] | None:
    """버퍼 ID의 화면 위치. OUT_* = 상차점, Q_* = 하차점(없으면 그 공정 설비 왼쪽)."""
    nodes = lay["nodes"]
    if bid.startswith("OUT_"):
        return node_xy(lay, bid)
    sid = bid[2:]
    if sid in nodes:
        return tuple(nodes[sid])
    if f"{sid}_IN" in nodes:
        return tuple(nodes[f"{sid}_IN"])
    ms = [v for k, v in nodes.items() if k.rsplit("_", 1)[0] == sid and k.rsplit("_", 1)[1].isdigit()]
    if not ms:
        return None
    return (min(v[0] for v in ms) - 3.5, sum(v[1] for v in ms) / len(ms))


def _walk_positions(walks: pd.DataFrame, lay: dict, home: tuple, ts: np.ndarray) -> list[tuple]:
    """OP_WALK_START(출발)→OP_ARRIVE(도착) 사이를 직선 보간한 작업자 위치."""
    if walks.empty:
        return [home] * len(ts)
    t0, t1 = walks.time_s.to_numpy(), walks.t_arrive.to_numpy()
    a = [node_xy(lay, str(n)) or home for n in walks.node_from]
    b = [node_xy(lay, str(n)) or home for n in walks.node_to]
    out = []
    for t, i in zip(ts, np.searchsorted(t0, ts, side="right") - 1):
        if i < 0:
            out.append(home)
        elif t < t1[i]:
            f = (t - t0[i]) / max(t1[i] - t0[i], 1e-9)
            out.append((a[i][0] + f * (b[i][0] - a[i][0]), a[i][1] + f * (b[i][1] - a[i][1])))
        else:
            out.append(b[i])
    return out


@st.cache_data(show_spinner="애니메이션 프레임 계산 중...")
def animation_figure(run_dir: str, start_h: float, dur_h: float, step_s: float, frame_ms: int) -> go.Figure:
    """2D 평면도를 Plotly 프레임으로 미리 계산해 브라우저에서 재생/일시정지한다 (서버 왕복 없음).

    표시: 버퍼 재공품(점), 가공 중 부품(설비 안 점, 품종군 색), 작업자 도보 이동, AGV 적재 수, 키트·출하 계기.
    """
    cfg = resolve_config(Path(run_dir) / "config_resolved.yaml")
    lay, nodes = cfg["layout"], cfg["layout"]["nodes"]
    ts = np.arange(start_h * 3600.0, (start_h + dur_h) * 3600.0 + 1e-9, step_s)

    ri = load_csv(str(Path(run_dir) / "resource_intervals.csv"),
                  ("resource_type", "resource_id", "state", "start_s", "end_s"))
    ev = load_csv(str(Path(run_dir) / "events.csv"),
                  ("time_s", "event_type", "group", "resource_id", "buffer_id", "buffer_count", "node_from", "node_to"))

    # 설비 상태 + 가공 중 부품의 품종군
    mach_ids = [k for k in nodes if "_" in k and k.rsplit("_", 1)[1].isdigit()]
    mi = ri[ri.resource_type == "machine"].sort_values("start_s")
    ps = ev[ev.event_type == "PROCESS_START"].sort_values("time_s")
    mstate, mgroup = {}, {}
    for mid in mach_ids:
        g = mi[mi.resource_id == mid]
        mstate[mid] = _step_lookup(g.start_s.to_numpy(), g.state.to_numpy(), ts, "")
        p = ps[ps.resource_id == mid]
        mgroup[mid] = _step_lookup(p.time_s.to_numpy(), p.group.astype(str).to_numpy(), ts, "")

    # 버퍼 재공품 수
    be = ev[ev.event_type.isin(["BUFFER_ENTER", "BUFFER_EXIT"])].sort_values("time_s", kind="stable")
    buf_ids = [b for b in sorted(be.buffer_id.dropna().unique()) if buffer_xy(lay, b)]
    bcount = {}
    for b in buf_ids:
        g = be[be.buffer_id == b]
        bcount[b] = _step_lookup(g.time_s.to_numpy(), g.buffer_count.fillna(0).astype(int).to_numpy(), ts, 0)
    slot_xy = []  # (버퍼, 슬롯 번호, x, y) — 점 개수를 프레임마다 같게 두고 크기로 켜고 끈다
    for b in buf_ids:
        bx, by = buffer_xy(lay, b)
        for k in range(BUF_SLOTS):
            slot_xy.append((b, k, bx - (BUF_COLS - 1) * BUF_GAP / 2 + (k % BUF_COLS) * BUF_GAP,
                            by - 2.2 - (k // BUF_COLS) * BUF_GAP))

    # 작업자: 도보 보간 + 상태 색
    wk = ev[ev.event_type == "OP_WALK_START"][["time_s", "resource_id", "node_from", "node_to"]].sort_values("time_s")
    ar = ev[ev.event_type == "OP_ARRIVE"][["time_s", "resource_id"]].sort_values("time_s")
    oi = ri[ri.resource_type == "operator"].sort_values("start_s")
    opos, ostate = {}, {}
    for op, oc in cfg["operators"]["skills"].items():
        w = wk[wk.resource_id == op].reset_index(drop=True)
        a = ar[ar.resource_id == op].time_s.to_numpy()
        w["t_arrive"] = a[np.minimum(np.searchsorted(a, w.time_s.to_numpy(), side="left"), len(a) - 1)] \
            if len(a) else w.time_s
        opos[op] = _walk_positions(w, lay, node_xy(lay, oc["home"]) or (0.0, 0.0), ts)
        g = oi[oi.resource_id == op]
        ostate[op] = _step_lookup(g.start_s.to_numpy(), g.state.to_numpy(), ts, "IDLE")

    # AGV 적재 수
    load_n = {}
    for agv, (_, trips) in agv_tables(run_dir).items():
        tr = trips.sort_values("t_depart")
        idx = np.searchsorted(tr.t_depart.to_numpy(), ts, side="right") - 1
        arr, n = tr.t_arrive_dest.to_numpy(), tr.n_items.to_numpy()
        load_n[agv] = [int(n[i]) if i >= 0 and t < arr[i] else 0 for t, i in zip(ts, idx)]

    kit, _ = occupancy_series(run_dir)
    kit_n = _step_lookup(kit.time_h.to_numpy() * 3600.0, kit.kit_occupied.to_numpy(), ts, 0) if not kit.empty \
        else [0] * len(ts)
    shipped = np.sort(ev.loc[ev.event_type == "ORDER_SHIPPED", "time_s"].to_numpy())
    ship_n = np.searchsorted(shipped, ts, side="right")
    k_slots = cfg["release"]["kit_slots"]
    kx, ky = nodes["KIT"]
    dx, dy = nodes["DOCK"]

    def frame_traces(i: int) -> list[go.Scatter]:
        t = ts[i]
        ms = {m: mstate[m][i] for m in mach_ids}
        busy = [m for m in mach_ids if ms[m] == "PROCESSING"]
        sizes = [9 if k < bcount[b][i] else 0 for b, k, _, _ in slot_xy]
        agv = agv_positions_at(run_dir, cfg, t)
        loads = [load_n.get(a, [0] * len(ts))[i] for a in agv]
        ops = list(opos)
        return [
            go.Scatter(x=[nodes[m][0] for m in mach_ids], y=[nodes[m][1] for m in mach_ids],
                       marker=dict(color=[STATE_COLORS.get(ms[m], "#457b9d") for m in mach_ids]),
                       hovertext=[f"{m}: {STATE_LABELS.get(ms[m], '')}" for m in mach_ids]),
            go.Scatter(x=[nodes[m][0] for m in mach_ids], y=[nodes[m][1] for m in mach_ids],
                       marker=dict(size=[8 if m in busy else 0 for m in mach_ids],
                                   color=[GROUP_COLORS.get(mgroup[m][i], "#ffffff") for m in mach_ids])),
            go.Scatter(x=[s[2] for s in slot_xy], y=[s[3] for s in slot_xy], marker=dict(size=sizes)),
            go.Scatter(x=[buffer_xy(lay, b)[0] + (BUF_COLS / 2 + 0.3) * BUF_GAP for b in buf_ids],
                       y=[buffer_xy(lay, b)[1] - 2.2 for b in buf_ids],
                       text=[str(bcount[b][i]) if bcount[b][i] else "" for b in buf_ids]),
            go.Scatter(x=[p[0] for p in agv.values()], y=[p[1] for p in agv.values()],
                       text=[f"{a} ▣{n}" if n else a for a, n in zip(agv, loads)],
                       marker=dict(color=["#f77f00" if n else "#ffd6a5" for n in loads],
                                   size=[18 if n else 13 for n in loads])),
            go.Scatter(x=[opos[o][i][0] + 1.0 for o in ops], y=[opos[o][i][1] - 1.0 for o in ops],
                       marker=dict(color=[OP_COLORS.get(ostate[o][i], "#3a0ca3") for o in ops]),
                       hovertext=[f"{o}: {OP_LABELS.get(ostate[o][i], ostate[o][i])}" for o in ops]),
            go.Scatter(x=[kx, dx], y=[ky + 6, dy + 6],
                       text=[f"키트 {kit_n[i]}/{k_slots}", f"출하 {ship_n[i]:,}대"]),
            go.Scatter(text=[f"t = {t / 3600:7.2f} h"]),
        ]

    fig = go.Figure()
    for a, b in LANES:
        if a in nodes and b in nodes:
            fig.add_trace(go.Scatter(x=[nodes[a][0], nodes[b][0]], y=[nodes[a][1], nodes[b][1]], mode="lines",
                                     line=dict(color="#ced4da", dash="dot"), hoverinfo="skip", showlegend=False))
    other = {k: v for k, v in nodes.items() if k not in mach_ids}
    fig.add_trace(go.Scatter(x=[v[0] for v in other.values()], y=[v[1] for v in other.values()], mode="markers+text",
                             text=list(other), textposition="bottom center", textfont=dict(size=9),
                             marker=dict(size=7, color="#6c757d"), hoverinfo="skip", showlegend=False))
    first = len(fig.data)
    f0 = frame_traces(0)
    styles = [
        dict(mode="markers+text", text=mach_ids, textposition="top center", textfont=dict(size=9),
             marker_symbol="square", marker_size=22, hoverinfo="text", name="설비(색=상태)"),
        dict(mode="markers", marker_symbol="circle", hoverinfo="skip", name="가공 중 부품"),
        dict(mode="markers", marker_color="#264653", hoverinfo="skip", name="버퍼 재공품"),
        dict(mode="text", textposition="middle right", textfont=dict(size=11, color="#264653"),
             hoverinfo="skip", showlegend=False),
        dict(mode="markers+text", marker_symbol="diamond", textposition="middle right", hoverinfo="text",
             name="AGV(진한색=적재)"),
        dict(mode="markers", marker_size=12, marker_line=dict(color="#ffffff", width=1), hoverinfo="text",
             name="작업자(색=상태)"),
        dict(mode="text", textfont=dict(size=14, color="#1d3557"), hoverinfo="skip", showlegend=False),
        dict(mode="text", x=[2.0], y=[46.0], textposition="middle right",
             textfont=dict(size=18, family="monospace", color="#1d3557"), hoverinfo="skip", showlegend=False),
    ]
    for tr, sty in zip(f0, styles):
        tr.update(**sty)
        fig.add_trace(tr)
    idx = list(range(first, first + len(f0)))
    fig.frames = [go.Frame(name=f"{t / 3600:.2f}", data=frame_traces(i), traces=idx) for i, t in enumerate(ts)]

    # 범례 견본: 설비 상태 · 품종군 · 작업자 상태
    for s in STATES:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"설비: {STATE_LABELS[s]}",
                                 marker=dict(size=11, symbol="square", color=STATE_COLORS[s])))
    for g, c in GROUP_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"부품: 품종군 {g}",
                                 marker=dict(size=9, color=c)))
    for s, c in OP_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"작업자: {OP_LABELS[s]}",
                                 marker=dict(size=11, color=c)))

    # redraw=False + 선형 전환 → 프레임 사이 위치가 부드럽게 이어진다
    play = dict(frame=dict(duration=frame_ms, redraw=False), transition=dict(duration=frame_ms, easing="linear"),
                fromcurrent=True, mode="immediate")
    seek = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")
    xs = [v[0] for v in nodes.values()]
    ys = [v[1] for v in nodes.values()] + [ys_ for _, _, _, ys_ in slot_xy]
    fig.update_layout(
        height=700, margin=dict(l=10, r=10, t=20, b=110), plot_bgcolor="#ffffff",
        xaxis=dict(range=[min(xs) - 4, max(xs) + 8], autorange=False, visible=False),
        yaxis=dict(range=[min(ys) - 2, max(ys) + 9], autorange=False, visible=False, scaleanchor="x"),
        legend=dict(orientation="v", x=1.01, y=1.0, font=dict(size=10)),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.04, xanchor="left", yanchor="top",
                          buttons=[dict(label="▶ 재생", method="animate", args=[None, play]),
                                   dict(label="⏸ 정지", method="animate", args=[[None], seek])])],
        sliders=[dict(x=0.16, y=-0.02, len=0.84, ticklen=0, font=dict(color="rgba(0,0,0,0)"),
                      currentvalue=dict(prefix="시각(h) ", font=dict(color="#1d3557")),
                      steps=[dict(label=f.name, method="animate", args=[[f.name], seek]) for f in fig.frames])])
    return fig


# ----------------------------------------------------------------------------- tabs
def tab_model() -> None:
    rev = st.selectbox("설계 Rev", list(REVS), index=len(REVS) - 1, key="model_rev",
                       format_func=lambda r: f"Rev.{r} ({REVS[r]})")
    try:
        cfg = load_rev(rev)
    except Exception as exc:  # 설정 파일 누락·오류
        st.error(f"설정을 읽지 못했습니다: {exc}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("설비", sum(int(s["machines"]) for s in cfg["stations"].values()))
    c2.metric("작업자", len(cfg["operators"]["skills"]))
    c3.metric("AGV", cfg["agv"]["count"])
    c4.metric("키트 슬롯 K / 하우징 s", f"{cfg['release']['kit_slots']} / {cfg['housing']['reorder_point']}")
    st.caption(f"수요 {cfg['demand']['rate_per_h']}대/h · 납기 도착+{cfg['demand']['due_offset_s'] / 3600:.0f}h · "
               f"워밍업 {cfg['run']['warmup_s'] / 3600:.0f}h + 관측 {cfg['run']['observation_s'] / 3600:.0f}h")
    st.subheader("레이아웃 평면도 (설명용 좌표)")
    st.plotly_chart(layout_figure(cfg), use_container_width=True)
    st.subheader("공정 그래프")
    st.plotly_chart(process_graph_figure(cfg), use_container_width=True)
    st.subheader("숙련 매트릭스")
    procs = [s for s, sc in cfg["stations"].items() if sc.get("operator", "none") != "none"]
    mat = pd.DataFrame({op: {p: ("●" if p in oc["skills"] else "") for p in procs}
                        for op, oc in cfg["operators"]["skills"].items()}).T
    st.dataframe(mat, use_container_width=True)
    st.subheader("정적 부하 계산 (가설, 시뮬레이션 결과 아님)")
    res = load_static(rev)
    cols = ["station", "machines", "mean_time_s", "cap_pure_per_h", "cap_setup_per_h", "cap_fail_per_h",
            "cap_effective_per_h", "util_at_demand", "rank", "note"]
    sdf = res["stations"]
    st.dataframe(sdf[[c for c in cols if c in sdf.columns]].round(3), use_container_width=True)
    h, k, ag, op = res["housing"], res["kit"], res["agv"], res["operators"]
    st.markdown(f"- 하우징 공급 상한: **{h.get('supply_cap_per_h', float('nan')):.1f}대/h** "
                f"(리드타임 중 수요 {h.get('demand_during_lead', 0):.0f}, s={h.get('reorder_point')})\n"
                f"- 키트 CONWIP 상한: {k['conwip_cap_per_h']:.1f}대/h (K={k['kit_slots']})\n"
                f"- AGV 필요 대수: 1개씩 {ag['agv_needed_batch1']:.2f} / 4개 묶음 {ag['agv_needed_batch4']:.2f} "
                f"(보유 {ag['count']}, 충전 가용률 {ag['availability']:.3f})\n"
                f"- 작업자 상시 점유: {op['busy_operators_at_demand']:.2f}명 / {op['operators']}명")


def tab_run() -> None:
    runs = list_full_runs()
    if not runs:
        no_results("full 상세 run이 없습니다.")
        return
    keys = list(runs)
    default = next((i for i, k in enumerate(keys) if "rev_C" in k), 0)  # 정상 가동 중인 최종안을 먼저 보여 준다
    key = st.selectbox("run (full 상세)", keys, index=default, key="run_sel")
    d = runs[key]
    m = load_metrics(str(d))
    try:
        cfg = resolve_config(d / "config_resolved.yaml")
    except Exception as exc:
        st.error(f"config_resolved.yaml을 읽지 못했습니다: {exc}")
        return
    f, tg = m["factory"], m.get("targets", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("수요 충족률", f"{f['demand_fulfil_ratio']:.3f}", "목표 달성" if tg.get("throughput_ok") else "미달")
    c2.metric("납기 충족률", f"{f['otd_rate']:.3f}", "목표 달성" if tg.get("otd_ok") else "미달")
    lt = f["lead_time_arrival_cohort"]["mean_lower_bound_s"]
    c3.metric("리드타임 하한", f"{lt / 3600:.1f} h", "목표 달성" if tg.get("leadtime_ok") else "미달")
    c4.metric("양품 출하", f"{f['good_rate_per_h']:.2f} 대/h")
    c5.metric("평균 WIP", f"{f['avg_wip']:.1f}")
    st.caption(f"상태 {m.get('status')} · 무결성 검사 {'전부 통과' if m.get('checks', {}).get('all_ok') else '실패 항목 있음'}"
               f" · 투입 정지(K 가득) {f['release_blocked_k_s']:,.0f}s · 결품 정지 {f['release_blocked_stockout_s']:,.0f}s")

    st.subheader("설비별 손실 분해 (관측시간 대비, 7범주)")
    st.plotly_chart(stacked_states(m["machines"], STATES, STATE_LABELS, STATE_COLORS), use_container_width=True)
    st.subheader("공정 손실 워터폴")
    stations = list(m["stations"])
    default = max(stations, key=lambda s: m["stations"][s].get("PROCESSING") or 0.0)
    sid = st.selectbox("공정 (기본: 가공 비율 최대)", stations, index=stations.index(default), key="wf_sid")
    st.plotly_chart(waterfall_figure(m, cfg, sid), use_container_width=True)
    st.caption("이론 능력 = 대수 × 3600 / 평균 처리시간 (정적 계산). 손실은 관측시간 비율만큼 능력을 뺀 근사다.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("작업자 상태")
        st.plotly_chart(stacked_states(m["operators"], OP_STATES, height=340), use_container_width=True)
    with c2:
        st.subheader("AGV 상태")
        st.plotly_chart(stacked_states(m["agvs"], AGV_STATES, height=340), use_container_width=True)

    st.subheader("키트 슬롯·버퍼 점유")
    kit, buf = occupancy_series(str(d))
    fig = go.Figure()
    if not kit.empty:
        fig.add_trace(go.Scatter(x=kit.time_h, y=kit.kit_occupied, mode="lines", line_shape="hv", name="키트 점유"))
        fig.add_hline(y=cfg["release"]["kit_slots"], line_dash="dash", annotation_text="K")
    bsel = st.multiselect("버퍼", sorted(buf.buffer_id.dropna().unique()) if not buf.empty else [],
                          default=[b for b in ("Q_S5",) if not buf.empty and b in set(buf.buffer_id)], key="bufsel")
    for b in bsel:
        s = buf[buf.buffer_id == b]
        fig.add_trace(go.Scatter(x=s.time_h, y=s.buffer_count, mode="lines", line_shape="hv", name=b))
    w0, w1 = m["factory"]["window_s"]
    fig.add_vrect(x0=w0 / 3600, x1=w1 / 3600, fillcolor="#e9ecef", opacity=0.3, line_width=0)
    fig.update_layout(height=340, xaxis_title="시간 (h)", yaxis_title="개수", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    horizon_h = float(m["factory"]["window_s"][1]) / 3600.0
    st.subheader("2D 애니메이션 재생")
    c1, c2, c3, c4 = st.columns(4)
    start_h = c1.number_input("시작 시각 (h)", 0.0, horizon_h - 0.5, min(100.0, horizon_h - 0.5), 0.5, key="anim_t0")
    dur_h = c2.selectbox("재생 구간 (h)", [0.5, 1.0, 2.0, 4.0], index=1, key="anim_dur")
    step_s = c3.selectbox("프레임 간격 (시뮬레이션 초)", [5, 10, 20, 30, 60], index=1, key="anim_step")
    frame_ms = c4.selectbox("프레임당 표시 시간 (ms)", [50, 100, 200], index=1, key="anim_ms")
    dur_h = min(dur_h, horizon_h - start_h)
    st.caption(f"프레임 {int(dur_h * 3600 / step_s) + 1}개 · 재생 속도 실제의 약 {step_s * 1000 / frame_ms:,.0f}배 "
               f"(구간 전체 약 {dur_h * 3600 / step_s * frame_ms / 1000:.0f}초)")
    st.plotly_chart(animation_figure(str(d), start_h, dur_h, float(step_s), int(frame_ms)), use_container_width=True)
    st.caption("설비 색 = 그 시각 상태(범례는 손실 분해와 같음). AGV는 운반 기록으로 보간, 작업자는 마지막 배정 설비에 표시.")

    with st.expander("특정 시각 정지 화면 (시간 슬라이더)"):
        t_h = st.slider("시각 (h)", 0.0, horizon_h, min(100.0, horizon_h), 0.25, key="play_t")
        t = t_h * 3600.0
        st.plotly_chart(layout_figure(cfg, machine_state_at(str(d), t), agv_positions_at(str(d), cfg, t),
                                      operator_positions_at(str(d), cfg, t)), use_container_width=True)


def candidate_table(exp_dir: Path) -> pd.DataFrame:
    plan = json.loads((exp_dir / "plan.json").read_text(encoding="utf-8"))
    paired = load_csv(str(exp_dir / "paired.csv"))
    sel_path = exp_dir / "selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    sel_csv = load_csv(str(exp_dir / "selection.csv"))
    elig = dict(zip(sel_csv.scenario, sel_csv.eligible)) if not sel_csv.empty else {}
    cands = plan.get("study_def", {}).get("candidates", {})
    rows = []
    for name, c in cands.items():
        row = {"후보": name, "변경": c.get("description", ""), "자원 유형": c.get("resource_class", "")}
        for metric, label in METRIC_LABELS.items():
            r = paired[(paired.scenario == name) & (paired.metric == metric)]
            if len(r):
                r = r.iloc[0]
                row[f"Δ{label}"] = f"{r.mean_d:+.4g} [{r.ci_low:.4g}, {r.ci_high:.4g}]"
        row["적격"] = "적격" if elig.get(name) else "부적격"
        row["판정"] = "채택" if sel.get("chosen") == name else "기각"
        rows.append(row)
    return pd.DataFrame(rows)


def targets_table(exp_dir: Path) -> pd.DataFrame:
    t = load_csv(str(exp_dir / "targets.csv"))
    plan = json.loads((exp_dir / "plan.json").read_text(encoding="utf-8"))
    revs = {k: v.get("rev") for k, v in plan.get("scenarios", {}).items()}
    out = pd.DataFrame({
        "설계": t.scenario.map(lambda s: f"Rev.{revs.get(s, s)}"),
        "조건": t.condition,
        "수요 충족률": [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in
                    zip(t.throughput_mean, t.throughput_ci_low, t.throughput_ci_high)],
        "납기 충족률": [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(t.otd_mean, t.otd_ci_low, t.otd_ci_high)],
        "리드타임 하한(s)": [f"{a:,.0f} [{b:,.0f}, {c:,.0f}]" for a, b, c in
                       zip(t.leadtime_mean, t.leadtime_ci_low, t.leadtime_ci_high)],
        "평균 기준": t.all_ok.map({True: "만족", False: "미달"}),
        "CI 경계 기준": t.all_ci_ok.map({True: "만족", False: "미달"}),
    })
    return out


def tab_design() -> None:
    confirm1, confirm2 = latest_experiment("final_confirm_"), latest_experiment("p2_final_confirm2_")
    if confirm1 is None and confirm2 is None:
        no_results("확인 실험(final_confirm / p2_final_confirm2) 결과가 없습니다.")
        return
    st.subheader("설계 개정 지표 추이 (확인 시드 평균, 95% CI)")
    points = []
    for exp, cond_map in ((confirm1, {("BASE", "base"): "Rev.A @18", ("FINAL", "base"): "Rev.B @18"}),
                          (confirm2, {("BASE", "p2_base"): "Rev.B @20.7", ("FINAL", "p2_base"): "Rev.C @20.7"})):
        if exp is None:
            continue
        t = load_csv(str(exp / "targets.csv"))
        for (sc, cond), label in cond_map.items():
            r = t[(t.scenario == sc) & (t.condition == cond)]
            if len(r):
                points.append((label, exp.name, r.iloc[0]))
    if points:
        c1, c2, c3 = st.columns(3)
        for col, (key, title, target) in zip((c1, c2, c3), (("throughput", "수요 충족률", 0.98),
                                                             ("otd", "납기 충족률", 0.95),
                                                             ("leadtime", "리드타임 하한 (h)", 16.0))):
            scale = 1 / 3600 if key == "leadtime" else 1.0
            y = np.array([p[2][f"{key}_mean"] for p in points]) * scale
            lo = np.array([p[2][f"{key}_ci_low"] for p in points]) * scale
            hi = np.array([p[2][f"{key}_ci_high"] for p in points]) * scale
            fig = go.Figure(go.Bar(x=[p[0] for p in points], y=y,
                                   error_y=dict(type="data", array=hi - y, arrayminus=y - lo)))
            fig.add_hline(y=target, line_dash="dash", annotation_text="목표")
            fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
            col.plotly_chart(fig, use_container_width=True)
        st.caption("실험: " + ", ".join(sorted({p[1] for p in points})))

    st.subheader("개정별 후보 비교 (탐색 시드 짝 차이, 95% CI)")
    for prefix, title in (("rev1_explore_", "1단계 개정 1: Rev.A → Rev.B (수요 18대/h)"),
                          ("p2_rev1_explore2_", "2단계 개정 1: Rev.B → Rev.C (수요 20.7대/h)")):
        exp = latest_experiment(prefix)
        if exp is None:
            continue
        st.markdown(f"**{title}** — `{exp.name}`")
        st.dataframe(candidate_table(exp), use_container_width=True, hide_index=True)
    st.caption(f"CI가 0을 포함하면 '{NO_CLEAR_DIFF}'. 채택 = 적격 후보 중 변경 용이성 순위 최소, 동률이면 primary 개선 최대.")

    st.subheader("최종 확인: 조건별 목표 판정")
    for exp, title in ((confirm1, "1단계 확인 (시드 1000~1029)"), (confirm2, "2단계 확인 (시드 2000~2029)")):
        if exp is None:
            continue
        st.markdown(f"**{title}** — `{exp.name}`")
        st.dataframe(targets_table(exp), use_container_width=True, hide_index=True)
    st.caption("목표를 못 지키는 조건은 '설계 적용 범위 밖'으로 보고한다 (docs/v2/design_log.md).")


# ----------------------------------------------------------------------------- main
st.warning(BANNER)
st.title("모터 공장 v2 공정설계")
t1, t2, t3 = st.tabs(["모델·실행", "한 실행의 진단", "설계 개정 비교"])
with t1:
    tab_model()
with t2:
    tab_run()
with t3:
    tab_design()

"""Streamlit 2D 분석 화면 (3개 탭). 엔진 로직은 line_sim 패키지에만 있다.

실행: python -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import line_sim  # noqa: F401
except ImportError:  # 설치 없이 실행할 때의 fallback
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from line_sim.config import load_yaml, nominal_capacity, product_types, validate
from line_sim.engine import run_simulation
from line_sim.metrics import (
    CSV_ENCODING, buffer_series, cumulative_series, load_run_dir, snapshot_at, wip_step_series,
)
from line_sim.run import write_run_dir

RESULTS = ROOT / "results"
BASELINE = ROOT / "configs" / "baseline.yaml"
BANNER = "가상 생산라인 / 입력값은 합성 가정 / 실제 공장 성능 검증 아님"
NO_CLEAR_DIFF = "이번 실험으로 명확한 차이를 확인하지 못함"
STATE_LABELS = {"PROCESSING": "가공", "SETUP": "준비", "BLOCKED": "막힘", "STARVED": "무작업"}
STATE_COLORS = {"PROCESSING": "#2a9d8f", "SETUP": "#e9c46a", "BLOCKED": "#e76f51", "STARVED": "#adb5bd"}
PRODUCT_COLORS = {"P1": "#264653", "P2": "#f4a261", "P3": "#8338ec"}

st.set_page_config(page_title="생산라인 시뮬레이션", layout="wide")


# ----------------------------------------------------------------------------- helpers
def fmt_window(cfg) -> str:
    w0, w1 = cfg["run"]["observation_window_s"]
    return f"관측창 {w0:,.0f}–{w1:,.0f}초"


def layout_positions(cfg) -> dict[str, tuple[float, float]]:
    """Logical (not physical) layout: x = route depth from the first station, fail branches below."""
    stations = cfg["line"]["stations"]
    pos: dict[str, tuple[float, float]] = {}
    first = cfg["line"]["first_station"]
    queue = [(first, 0, 0.0)]
    while queue:
        sid, depth, y = queue.pop(0)
        if sid in pos:
            continue
        pos[sid] = (float(depth), y)
        if sid not in stations:
            continue
        route = stations[sid]["route"]
        if route.get("type", "next") == "next":
            queue.append((route["next"], depth + 1, y))
        else:
            queue.append((route["on_pass"], depth + 1 + (1 if route["on_pass"] in stations else 0), y))
            queue.append((route["on_fail"], depth + 1, y - 1.0))
    return pos


def process_diagram(cfg, states: dict[str, str] | None = None, title: str = "") -> go.Figure:
    stations = cfg["line"]["stations"]
    pos = layout_positions(cfg)
    fig = go.Figure()
    for sid, st_cfg in stations.items():
        route = st_cfg["route"]
        edges = ([(route["next"], "")] if route.get("type", "next") == "next"
                 else [(route["on_pass"], "합격" if route["type"] == "inspect" else "회복"),
                       (route["on_fail"], "불합격" if route["type"] == "inspect" else "실패")])
        for tgt, label in edges:
            if tgt not in pos:
                continue
            (x0, y0), (x1, y1) = pos[sid], pos[tgt]
            fig.add_annotation(x=x1 - 0.28, y=y1, ax=x0 + 0.28, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1.5,
                               arrowcolor="#c1121f" if label in ("불합격", "실패") else "#495057")
            if label:
                fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.12, text=label, showarrow=False,
                                   font=dict(size=11))
    xs, ys, texts, colors = [], [], [], []
    for node, (x, y) in pos.items():
        xs.append(x)
        ys.append(y)
        if node in stations:
            s = stations[node]
            cap = s["input_buffer"]["capacity"]
            text = (f"<b>{node}</b><br>{s['machines']}대<br>{s['input_buffer']['id']}"
                    f"({'∞' if cap is None else cap})")
            if states:
                text += "<br>" + ", ".join(f"{STATE_LABELS.get(states.get(f'{node}_{k}'), '-')}"
                                           for k in range(1, s["machines"] + 1))
            colors.append("#dee2e6")
        else:
            text = "<b>양품 출하</b>" if node == "SHIP" else "<b>폐기</b>"
            colors.append("#d8f3dc" if node == "SHIP" else "#ffccd5")
        texts.append(text)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textposition="middle center",
                             marker=dict(symbol="square", size=95, color=colors, line=dict(color="#495057")),
                             hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(visible=False, range=[-0.6, max(x for x, _ in pos.values()) + 0.6]),
                      yaxis=dict(visible=False, range=[min(y for _, y in pos.values()) - 0.6, 0.6]))
    return fig


def list_run_dirs() -> list[Path]:
    if not RESULTS.exists():
        return []
    dirs = [p.parent for p in RESULTS.rglob("metrics.json")]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def list_experiments() -> list[Path]:
    if not RESULTS.exists():
        return []
    return sorted((p.parent for p in RESULTS.glob("*/experiment.json")), key=lambda d: d.stat().st_mtime,
                  reverse=True)


@st.cache_data(show_spinner=False)
def cached_run(path: str, mtime: float):
    rd = load_run_dir(path)
    metrics = json.loads((Path(path) / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8"))
    return rd, metrics, meta


@st.cache_data(show_spinner=False)
def cached_experiment(path: str, mtime: float):
    p = Path(path)
    exp = json.loads((p / "experiment.json").read_text(encoding="utf-8"))
    def read(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(p / name, encoding=CSV_ENCODING)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    return exp, read("summary.csv"), read("paired.csv"), read("summary_by_product.csv"), read("runs_index.csv")


def experiment_window_text(path: Path, index: pd.DataFrame) -> str:
    """Observation window read from the experiment's own resolved run configs."""
    windows = set()
    ok = index[index.status != "failed"].drop_duplicates(["scenario", "condition"]) if not index.empty else index
    for rel in (ok["run_dir"] if not ok.empty else []):
        f = path / rel / "config_resolved.yaml"
        if f.exists():
            windows.add(fmt_window(load_yaml(f)))
    if not windows:
        return "관측창 미상"
    return " / ".join(sorted(windows))


def pct(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------- tab 1
def tab_model_run():
    base = load_yaml(BASELINE)
    cfg = copy.deepcopy(base)
    products = product_types(base)
    stations = list(base["line"]["stations"])

    st.subheader("품종별 작업시간 (초/제품) — 합성 가정")
    times_df = pd.DataFrame({sid: [base["line"]["stations"][sid]["times_s"][p] for p in products]
                             for sid in stations}, index=products)
    times_df["불합격률"] = [base["products"][p]["fail_rate"] for p in products]
    times_df["회복확률"] = [base["products"][p]["recover_prob"] for p in products]
    edited = st.data_editor(times_df, key="times_editor")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**주문 수(개)**")
        for p in products:
            cfg["orders"]["counts"][p] = int(st.number_input(p, value=base["orders"]["counts"][p], step=1,
                                                             key=f"cnt_{p}"))
    with c2:
        st.markdown("**설비 대수 / 입력 버퍼 용량**")
        for sid in stations:
            s = base["line"]["stations"][sid]
            a, b, c = st.columns([2, 2, 1])
            cfg["line"]["stations"][sid]["machines"] = int(a.number_input(f"{sid} 대수", value=s["machines"], step=1,
                                                                          key=f"m_{sid}"))
            base_cap = s["input_buffer"]["capacity"]
            inf_cap = c.checkbox("무제한", value=base_cap is None, key=f"binf_{sid}")
            cap_val = int(b.number_input(f"{s['input_buffer']['id']} 용량", value=8 if base_cap is None else base_cap,
                                         step=1, key=f"b_{sid}", disabled=inf_cap))
            cfg["line"]["stations"][sid]["input_buffer"]["capacity"] = None if inf_cap else cap_val
    with c3:
        st.markdown("**투입·실행**")
        cfg["release"]["min_interval_s"] = float(st.number_input("최소 투입 간격(초)",
                                                                 value=float(base["release"]["min_interval_s"])))
        base_wl = base["release"]["wip_limit"]
        unlimited = st.checkbox("WIP 상한 없음", value=base_wl is None)
        wl = int(st.number_input("전체 WIP 상한(개)", value=40 if base_wl is None else base_wl, step=1,
                                 disabled=unlimited))
        cfg["release"]["wip_limit"] = None if unlimited else wl
        policy = st.selectbox("투입 순서", ["random_permutation", "grouped"],
                              format_func=lambda x: "시드 기반 무작위 순열" if x == "random_permutation"
                              else "품종 묶음 P1→P2→P3")
        cfg["release"]["sequence"] = ({"policy": policy} if policy == "random_permutation"
                                      else {"policy": "grouped", "order": products})
        cfg["stochastic"]["enabled"] = st.checkbox("확률 모드(로그정규)", value=True)
        cfg["stochastic"]["cv"] = float(st.number_input("처리시간 CV", value=float(base["stochastic"]["cv"]),
                                                        step=0.05))
        seed = int(st.number_input("시드", value=100, step=1))

    for sid in stations:
        for p in products:
            cfg["line"]["stations"][sid]["times_s"][p] = float(edited.loc[p, sid])
    for p in products:
        cfg["products"][p]["fail_rate"] = float(edited.loc[p, "불합격률"])
        cfg["products"][p]["recover_prob"] = float(edited.loc[p, "회복확률"])

    errors = validate(cfg)
    if errors:
        st.error("설정 오류 — 실행할 수 없음:\n\n" + "\n".join(f"- {e}" for e in errors))

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(process_diagram(cfg, title="공정 연결도 (논리도, 실제 배치 도면 아님)"),
                        use_container_width=True)
    with right:
        if not errors:
            cap = nominal_capacity(cfg)
            st.markdown("**장기 처리능력 참고값 (개/시간)** — 준비시간·변동·버퍼·불량 제외, 품종 비율 가중")
            st.dataframe(pd.DataFrame({"처리능력(개/시간)": {k: None if v is None else round(v, 2)
                                                         for k, v in cap.items()}}))

    if st.button("단일 실행", type="primary", disabled=bool(errors)):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out = RESULTS / "ui_adhoc" / ts / f"seed_{seed}"
        try:
            with st.spinner("시뮬레이션 실행 중..."):
                result = run_simulation(cfg, seed, run_id=f"ui_{ts}_seed_{seed}")
                write_run_dir(result, out, scenario="ui_adhoc", command="streamlit app/dashboard.py")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            st.error(f"실행 실패: {exc!r}")
        else:
            st.session_state["last_run_dir"] = str(out)
            st.success(f"저장: {out.relative_to(ROOT).as_posix()} (상태: {result.status})")
    if st.session_state.get("last_run_dir"):
        st.info("UI 임의 실행 — 단일 시드, 보고서 프로토콜 아님. '한 실행의 진단' 탭에서 확인하세요. "
                "이 화면의 숫자만으로 최종 결론을 내리지 않습니다.")


# ----------------------------------------------------------------------------- tab 2
def tab_run_diagnosis():
    dirs = list_run_dirs()
    if not dirs:
        st.info("저장된 run이 없습니다. 탭 1에서 실행하거나 CLI로 실행하세요.")
        return
    labels = [d.relative_to(RESULTS).as_posix() for d in dirs]
    default = 0
    last = st.session_state.get("last_run_dir")
    if last and Path(last) in dirs:
        default = dirs.index(Path(last))
    choice = st.selectbox("run 디렉토리 (results/ 기준)", labels, index=default)
    run_dir = RESULTS / choice
    rd, metrics, meta = cached_run(str(run_dir), (run_dir / "metrics.json").stat().st_mtime)
    cfg = rd.config
    shift, full = metrics["shift"], metrics["full_run"]
    win = fmt_window(cfg)

    st.caption(f"run_id: {meta.get('run_id')} · 시드 {meta.get('seed')} · 상태 {meta.get('status')} · "
               f"config_hash {str(meta.get('config_hash'))[:12]} · experiment {meta.get('experiment_id')}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"교대 내 양품 / 주문 ({win})", f"{shift['good_shipped']} / {shift['orders_total']}")
    k2.metric("전체 납기 충족률", pct(shift["overall_otd_rate"]))
    k3.metric(f"시간평균 WIP (개, {win})", f"{shift['avg_wip']:.2f}")
    ct = full["completion_time_s"]
    k4.metric("전체 주문 종결시간 (초)", "null (미완료)" if ct is None else f"{ct:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**품종별 납기 충족률**")
        st.dataframe(pd.DataFrame([{"품종": p, "주문": v["orders"], "교대 내 양품": v["shipped"],
                                    "납기 충족률": pct(v["otd_rate"])} for p, v in shift["by_product"].items()]),
                     hide_index=True)
    with c2:
        u = shift["unshipped"]
        ft = shift["completed_flow_time"]
        inc = shift["incomplete"]
        st.markdown("**미출하 분해 / 체류시간(완료품 기준)**")
        st.dataframe(pd.DataFrame([
            {"항목": "미출하 합계", "값": u["total"]}, {"항목": "미투입", "값": u["unreleased"]},
            {"항목": "WIP", "값": u["wip"]}, {"항목": "폐기", "값": u["scrapped"]},
            {"항목": f"완료품 라인 체류 평균(초, n={ft['n']})", "값": None if ft["mean_s"] is None else round(ft["mean_s"], 1)},
            {"항목": f"미완료 {inc['n']}개 평균 나이(초)", "값": None if inc["age_mean_s"] is None else round(inc["age_mean_s"], 1)},
        ]), hide_index=True)

    ev = rd.events
    w0, w1 = cfg["run"]["observation_window_s"]
    sim_end = rd.sim_end_s
    # Cumulative shipments & WIP
    ship = cumulative_series(ev, "JOB_SHIPPED")
    wip = wip_step_series(ev)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, *ship.time_s, sim_end], y=[0, *ship.value, ship.value.iloc[-1] if len(ship) else 0],
                             mode="lines", line_shape="hv", name="누적 양품 출하(개)"))
    fig.add_trace(go.Scatter(x=[0.0, *wip.time_s, sim_end], y=[0, *wip.value, wip.value.iloc[-1] if len(wip) else 0],
                             mode="lines", line_shape="hv", name="WIP(개)", yaxis="y2"))
    fig.add_vline(x=w1, line_dash="dash", annotation_text="교대 종료")
    fig.update_layout(title=f"누적 양품 출하와 WIP (초, 전체 실행 0–{sim_end:,.0f}초, 이벤트 기반 계단선)",
                      xaxis_title="시각(초)", yaxis_title="누적 양품(개)",
                      yaxis2=dict(title="WIP(개)", overlaying="y", side="right"), height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Machine state ratios (DOWN not shown: breakdown feature not implemented)
    mach = shift["machines"]
    fig = go.Figure()
    for state, key in (("PROCESSING", "processing"), ("SETUP", "setup"), ("BLOCKED", "blocked"),
                       ("STARVED", "starved")):
        fig.add_trace(go.Bar(y=list(mach), x=[mach[m][key] * 100 for m in mach], orientation="h",
                             name=STATE_LABELS[state], marker_color=STATE_COLORS[state]))
    fig.update_layout(barmode="stack", title=f"설비 상태 비율 (%, {win})", xaxis_title="비율(%)", height=360,
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("무작업(STARVED)에는 주문이 모두 끝난 뒤의 무작업 시간도 포함되므로 이 지표만으로 공급 문제라고 단정하지 않습니다.")

    # Buffer occupancy
    fig = go.Figure()
    for sid, s in cfg["line"]["stations"].items():
        bid = s["input_buffer"]["id"]
        bs = buffer_series(ev, bid)
        fig.add_trace(go.Scatter(x=[0.0, *bs.time_s, sim_end], y=[0, *bs.value, bs.value.iloc[-1] if len(bs) else 0],
                                 mode="lines", line_shape="hv", name=f"{bid} (용량 {s['input_buffer']['capacity']})"))
    fig.update_layout(title=f"버퍼 점유 (개, 전체 실행 0–{sim_end:,.0f}초) — 표: 시간평균·최대({win})",
                      xaxis_title="시각(초)", yaxis_title="대기 제품 수(개)", height=380)
    st.plotly_chart(fig, use_container_width=True)
    bsel = st.selectbox("용량선과 함께 볼 버퍼", [s["input_buffer"]["id"] for s in cfg["line"]["stations"].values()])
    cap = next(s["input_buffer"]["capacity"] for s in cfg["line"]["stations"].values() if s["input_buffer"]["id"] == bsel)
    bs = buffer_series(ev, bsel)
    f2 = go.Figure(go.Scatter(x=[0.0, *bs.time_s, sim_end], y=[0, *bs.value, bs.value.iloc[-1] if len(bs) else 0],
                              mode="lines", line_shape="hv", name=bsel))
    if cap is not None:
        f2.add_hline(y=cap, line_dash="dash", line_color="#c1121f", annotation_text=f"용량 {cap}")
    f2.update_layout(title=f"{bsel} 점유와 용량선 (개, 전체 실행 0–{sim_end:,.0f}초)", xaxis_title="시각(초)",
                     yaxis_title="대기 제품 수(개)", height=300)
    st.plotly_chart(f2, use_container_width=True)
    st.dataframe(pd.DataFrame({"시간평균 점유(개)": shift["buffer_avg"], "최대 점유(개)": shift["buffer_max"]}).round(3))

    # Gantt for a subset of jobs
    jobs = rd.jobs.sort_values("release_seq")
    max_show = min(60, len(jobs))
    if max_show >= 2:
        n_show = st.slider("Gantt에 표시할 Job 수 (투입 순서 앞부분)", 1, max_show, min(15, max_show))
    else:
        n_show = max_show
    sel_ids = jobs.job_id.head(n_show).tolist()
    iv = rd.intervals[rd.intervals.job_id.isin(sel_ids) & rd.intervals.state.isin(["SETUP", "PROCESSING", "BLOCKED"])]
    fig = go.Figure()
    for state in ("SETUP", "PROCESSING", "BLOCKED"):
        part = iv[iv.state == state]
        fig.add_trace(go.Bar(y=[f"Job {int(j)}" for j in part.job_id], x=part.duration_s, base=part.start_s,
                             orientation="h", name=STATE_LABELS[state], marker_color=STATE_COLORS[state],
                             text=part.machine_id, hovertemplate="%{text}<br>%{base:.1f}–%{x:.1f}초<extra></extra>"))
    fig.update_layout(barmode="overlay", title="Job별 설비 점유 Gantt (초, 버퍼 대기는 공백)", xaxis_title="시각(초)",
                      height=max(300, 22 * n_show), yaxis=dict(autorange="reversed",
                                                               categoryorder="array",
                                                               categoryarray=[f"Job {j}" for j in sel_ids]))
    st.plotly_chart(fig, use_container_width=True)

    jid = st.selectbox("Job 이력 보기", jobs.job_id.tolist())
    st.dataframe(ev[ev.job_id == jid].drop(columns=["run_id"]), hide_index=True)
    with st.expander("jobs.csv 행"):
        st.dataframe(jobs[jobs.job_id == jid].T.astype(str))

    # Replay via time slider (state reconstructed from the event log, KPI unaffected)
    st.markdown("**시간 슬라이더 재생** — 이벤트 로그에서 해당 시각 직후 상태를 복원")
    t = st.slider("시각(초)", 0.0, float(sim_end), float(min(w1, sim_end) / 2), step=10.0)
    snap = snapshot_at(ev, t)
    states = {m: v["state"] for m, v in snap["machines"].items()}
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("미투입", snap["unreleased"])
    s2.metric("WIP", snap["wip"])
    s3.metric("누적 양품", snap["shipped"])
    s4.metric("누적 폐기", snap["scrapped"])
    st.plotly_chart(process_diagram(cfg, states, title=f"t = {t:,.0f}초 공정 상태 (논리도)"), use_container_width=True)
    st.dataframe(pd.DataFrame([{"버퍼": b, "점유": c} for b, c in snap["buffers"].items()]), hide_index=True)


# ----------------------------------------------------------------------------- tab 3
def tab_experiments():
    exps = list_experiments()
    if not exps:
        st.info("저장된 실험이 없습니다. `python -m line_sim.experiments --matrix configs/experiments.yaml "
                "--phase discovery`를 먼저 실행하세요.")
        return
    choice = st.selectbox("실험 (results/<experiment_id>)", [e.name for e in exps])
    path = RESULTS / choice
    exp, summary, paired, by_product, index = cached_experiment(str(path), (path / "experiment.json").stat().st_mtime)
    seeds = exp.get("seeds") or []
    seed_text = f"{seeds[0]}–{seeds[-1]} ({len(seeds)}개)" if seeds else "없음"
    st.caption(f"experiment ID: **{exp['experiment_id']}** · phase {exp['phase']} · matrix hash "
               f"{exp['matrix_hash'][:12]} · git {str(exp.get('git_commit'))[:10]} (dirty={exp.get('git_dirty')}) · "
               f"시드 {seed_text} · 저장 결과 읽기 전용(재실행 안 함)")

    n_fail = int((index["status"] == "failed").sum()) if not index.empty else 0
    n_inc = int((index["status"] == "incomplete_at_limit").sum()) if not index.empty else 0
    if n_fail:
        st.error(f"실패 run {n_fail}개 — 통계에서 제외되었고 n_failed로 표시됩니다.")
        st.dataframe(index[index.status == "failed"][["scenario", "condition", "seed", "error_message"]])
    if n_inc:
        st.warning(f"제한시간 미완료 run {n_inc}개 — 전체 종결 KPI는 null(n_null)로 집계됩니다.")

    st.markdown("**시나리오와 물리적 투입 차이**")
    st.dataframe(pd.DataFrame([{"시나리오": k, "설명": v.get("description"), "물리적 차이": v.get("physical_delta")}
                               for k, v in exp["scenarios"].items()]), hide_index=True)

    if summary.empty:
        st.info("요약 통계가 없습니다.")
        return
    conditions = list(dict.fromkeys(summary.condition))
    metrics = list(dict.fromkeys(summary.metric))
    c1, c2 = st.columns(2)
    cond = c1.selectbox("조건", conditions)
    metric = c2.selectbox("지표", metrics, index=metrics.index("shift.good_shipped") if "shift.good_shipped" in metrics else 0)
    window_text = experiment_window_text(path, index)
    scope = window_text if metric.startswith("shift.") else "전체 실행"
    sel = summary[(summary.condition == cond) & (summary.metric == metric)]
    fig_main = go.Figure(go.Scatter(
        x=sel.scenario, y=sel["mean"], mode="markers",
        error_y=dict(type="data", symmetric=False, array=(sel.ci_high - sel["mean"]).fillna(0),
                     arrayminus=(sel["mean"] - sel.ci_low).fillna(0)), marker=dict(size=11)))
    fig_main.update_layout(title=f"{metric}: 평균 ± 95% t CI (시드별 1회 실행 단위, {scope}, 조건 {cond})",
                           xaxis_title="시나리오", yaxis_title=metric, height=380)
    st.plotly_chart(fig_main, use_container_width=True)
    st.dataframe(sel, hide_index=True)

    if not paired.empty:
        st.markdown(f"**짝 비교 (시드별 d = 후보 − {exp.get('baseline_scenario', 'B0')})**")
        psel = paired[(paired.condition == cond) & (paired.metric == metric)]
        st.dataframe(psel, hide_index=True)
        if (psel.ci_includes_zero == True).any():  # noqa: E712
            st.caption(f"CI가 0을 포함하는 경우: {NO_CLEAR_DIFF}. 효과가 정확히 0이라는 뜻은 아닙니다.")

    fig_prod = None
    if not by_product.empty:
        bsel = by_product[by_product.condition == cond]
        fig_prod = go.Figure()
        for p in dict.fromkeys(bsel.product_type):
            part = bsel[bsel.product_type == p]
            fig_prod.add_trace(go.Bar(x=part.scenario, y=part["mean"] * 100, name=p,
                                      marker_color=PRODUCT_COLORS.get(p),
                                      error_y=dict(type="data", symmetric=False,
                                                   array=((part.ci_high - part["mean"]) * 100).fillna(0),
                                                   arrayminus=((part["mean"] - part.ci_low) * 100).fillna(0))))
        fig_prod.update_layout(barmode="group", title=f"품종별 납기 충족률 (%, {window_text}, 평균 ± 95% CI, 조건 {cond})",
                               yaxis_title="납기 충족률(%)", height=380)
        st.plotly_chart(fig_prod, use_container_width=True)

    fig_cond = None
    if len(conditions) > 1 and not paired.empty:
        psel = paired[paired.metric == metric]
        fig_cond = go.Figure()
        for sc in dict.fromkeys(psel.scenario):
            part = psel[psel.scenario == sc]
            fig_cond.add_trace(go.Scatter(x=part.condition, y=part.mean_d, mode="markers", name=sc,
                                          error_y=dict(type="data", symmetric=False,
                                                       array=(part.ci_high - part.mean_d).fillna(0),
                                                       arrayminus=(part.mean_d - part.ci_low).fillna(0))))
        fig_cond.add_hline(y=0, line_dash="dot")
        fig_cond.update_layout(title=f"조건별 후보 효과 d ({metric}, 평균 ± 95% CI, {scope})",
                               xaxis_title="조건", yaxis_title="d", height=360)
        st.plotly_chart(fig_cond, use_container_width=True)
        hl = exp["conditions"].get("high_load_mix", {})
        if hl.get("comparable_to_base") is False:
            st.caption("high_load_mix는 주문 집합이 달라 기본 조건과 직접 짝 비교하지 않습니다(같은 조건 안 B0 대비만).")

    st.markdown("**내보내기**")
    d1, d2, d3, d4 = st.columns(4)
    for col, name in ((d1, "summary.csv"), (d2, "paired.csv"), (d3, "summary_by_product.csv")):
        f = path / name
        col.download_button(name, data=f.read_bytes(), file_name=f"{exp['experiment_id']}_{name}", mime="text/csv")
    figs = [fig_main] + [f for f in (fig_prod, fig_cond) if f is not None]
    html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs))
    d4.download_button("그림 HTML", data=f"<html><head><meta charset='utf-8'></head><body>"
                       f"<p>experiment ID: {exp['experiment_id']}</p>{html}</body></html>",
                       file_name=f"{exp['experiment_id']}_figures.html", mime="text/html")


# ----------------------------------------------------------------------------- page
st.title("3품종 생산라인 시뮬레이션")
st.warning(BANNER)
t1, t2, t3 = st.tabs(["모델·실행", "한 실행의 진단", "실험 비교"])
with t1:
    tab_model_run()
with t2:
    tab_run_diagnosis()
with t3:
    tab_experiments()

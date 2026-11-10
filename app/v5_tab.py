"""대시보드 'v5 통합' 탭 — 저장된 eval 결과만 읽는다 (S1 비전 → S2 MuJoCo 파지 → S3 라인)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MUJO = ROOT.parents[1] / "mujo"
COND_LABEL = {"base": "기준", "n01": "적재 1", "n10": "적재 10", "n20": "적재 20", "noise0": "노이즈 0",
              "noise2": "노이즈 2 mm", "noise4": "노이즈 4 mm", "drop00": "결측 0 %", "drop15": "결측 15 %",
              "drop30": "결측 30 %", "side_light": "측면 조명"}
COLORS = {"B0": "#8d99ae", "B1": "#2a9d8f"}


def _latest(parent: Path, pattern: str) -> Path | None:
    c = sorted(p for p in parent.glob(pattern) if p.is_dir())
    return c[-1] if c else None


def _jsonl(p: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


def _s1(d: Path):
    st.subheader("S1 비전 — 하우징 bin, B0(DBSCAN) vs B1(하우징 전용 YOLO)")
    st.caption(f"`{d.relative_to(ROOT)}` · 조건마다 20장면 · TCP 허용치 ±2 mm / ±3°")
    s = pd.read_csv(d / "summary.csv")
    order = list(COND_LABEL)
    fig = go.Figure()
    for pl in ("B0", "B1"):
        g = s[s.pipeline == pl].set_index("condition").reindex(order)
        fig.add_bar(name=f"{pl} 올바른 집기", x=[COND_LABEL[c] for c in order], y=g.correct_pick, marker_color=COLORS[pl])
        fig.add_scatter(name=f"{pl} 오채택", x=[COND_LABEL[c] for c in order], y=g.false_accept, mode="markers",
                        marker=dict(symbol="x", size=11, color=COLORS[pl], line=dict(width=2)))
    fig.update_layout(barmode="group", height=380, yaxis_title="장면 수 (/20)", legend_orientation="h",
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    j = d / "s1_judgement.json"
    if j.exists():
        jj = json.loads(j.read_text(encoding="utf-8"))
        pa = jj["paired_all"]["correct_pick"]
        c1, c2, c3 = st.columns(3)
        c1.metric("올바른 집기 합계 B0 → B1", f"{int(s[s.pipeline=='B0'].correct_pick.sum())} → {int(s[s.pipeline=='B1'].correct_pick.sum())}",
                  help="220장면")
        c2.metric("짝 비교 (B1만 / B0만)", f"{pa['b_only']} / {pa['a_only']}", help=f"정확 McNemar p = {pa['p_exact']:.1e}")
        fa = jj["false_accept_types"]
        c3.metric("B1 오채택 유형", f"뒤집힘 {fa['B1']['flipped']} · 근처 {fa['B1']['near_tolerance']}")
    st.info("B1은 대부분 조건에서 B0보다 많이 집지만, 깊이 노이즈 σ 4 mm에서는 채택 16건이 모두 오채택이다 (깊이 검사 문턱값이 σ 1 mm 기준).")


def _s2(d: Path):
    st.subheader("S2 파지 — MuJoCo 접촉 파지 (추정 자세 이식)")
    st.caption(f"`mujo/{d.relative_to(MUJO).as_posix()}` · 하우징은 정답 회전으로 놓고, 팔은 추정 자세만 본다")
    r = _jsonl(d / "pick_result.jsonl")
    tab = r.groupby("group").success.agg(["sum", "count"]).rename(columns={"sum": "성공", "count": "시도"})
    st.dataframe(tab, use_container_width=True)
    w = r[r.group == "G-WRONG"].copy()
    if len(w):
        w["유형"] = w.E_trans_mm.map(lambda v: "위아래 뒤집힘" if v > 20 else "허용치 근처 (2–3 mm)")
        fig = go.Figure()
        for ok, col, name in ((True, "#2a9d8f", "성공"), (False, "#d62828", "실패")):
            g = w[w.success == ok]
            fig.add_scatter(x=g.grasp_point_err_mm, y=g.E_rot_deg, mode="markers", name=name,
                            marker=dict(size=10, color=col), text=g.reason)
        fig.update_layout(height=320, xaxis_title="파지점 오차 (mm)", yaxis_title="자세 회전 오차 (°)",
                          margin=dict(t=10, b=10), legend_orientation="h")
        st.markdown("**오채택 자세로 집었을 때** — 허용치 근처 오차는 모두 성공, 뒤집힘은 모두 실패")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("회전 오차 120°는 다리 3개 대칭에 의한 것으로 파지에 영향이 없다.")


def _s3(d: Path):
    st.subheader("S3 라인 — Rev.C + VP 셀, 1회 실패율 민감도")
    st.caption(f"`{d.relative_to(ROOT)}` · 확인 시드 3000–3029 · Rev.C(VP 없음)와 공통 난수 짝 비교")
    s = pd.read_csv(d / "summary.csv", encoding="utf-8-sig")
    sens = {f"VP_F{p:02d}": p for p in (1, 3, 5, 10, 20, 40, 60, 80)}
    c1, c2 = st.columns(2)
    for col, metric, title, ref in ((c1, "demand_fulfil_ratio", "수요 충족률", 0.98),
                                    (c2, "lead_p90_lb_h", "리드타임 P90 하한 (h)", None)):
        fig = go.Figure()
        for dem, color in (("d20.7", "#457b9d"), ("d23.8", "#e76f51")):
            g = s[(s.metric == metric) & (s.demand == dem) & s.scenario.isin(sens)].copy()
            g["x"] = g.scenario.map(sens)
            g = g.sort_values("x")
            fig.add_scatter(x=g.x, y=g["mean"], name=f"{dem[1:]}대/h", line=dict(color=color),
                            error_y=dict(type="data", symmetric=False, array=g.ci_high - g["mean"],
                                         arrayminus=g["mean"] - g.ci_low))
        if ref:
            fig.add_hline(y=ref, line_dash="dot", annotation_text="목표 0.98")
        fig.update_layout(height=320, xaxis_title="1회 실패율 (%)", yaxis_title=title, margin=dict(t=30, b=10),
                          title=title, legend_orientation="h")
        col.plotly_chart(fig, use_container_width=True)
    rows = []
    for sc, lab in (("BASE", "Rev.C (VP 없음)"), ("VP_MEAS", "VP-측정 (기준)"), ("VP_N20", "VP-적재20")):
        for dem in ("d20.7", "d23.8"):
            g = s[(s.scenario == sc) & (s.demand == dem)].set_index("metric")
            m = g.loc["demand_fulfil_ratio"]
            rows.append({"시나리오": lab, "수요(대/h)": dem[1:], "수요 충족률": f"{m['mean']:.4f} [{m.ci_low:.4f}, {m.ci_high:.4f}]",
                         "Rev.C 대비 리드타임 P90 (h)": "—" if sc == "BASE" else
                         f"{g.loc['lead_p90_lb_h','diff_mean']:+.2f} [{g.loc['lead_p90_lb_h','diff_ci_low']:+.2f}, {g.loc['lead_p90_lb_h','diff_ci_high']:+.2f}]"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    vp = d / "vp_summary.csv"
    if vp.exists():
        v = pd.read_csv(vp, encoding="utf-8-sig")
        v = v[v.scenario.isin(["VP_MEAS", "VP_N20", "VP_F40", "VP_F80"])][
            ["scenario", "demand", "vp_manual_rate", "vp_util", "fa_wait_operator"]]
        st.dataframe(v.rename(columns={"vp_manual_rate": "수동 처리 비율", "vp_util": "VP 이용률",
                                       "fa_wait_operator": "FA 작업자 대기"}).round(3),
                     use_container_width=True, hide_index=True)
    st.info("측정 조건의 VP는 Rev.C 목표를 유지한다. 23.8대/h에서 1회 실패율 80 %일 때 처음 목표(0.98)를 밑돈다 — "
            "원인은 VP 자체가 아니라 수동 처리로 작업자를 빼앗겨 조립(FA)이 기다리는 것.")


def tab_v5():
    st.markdown("빈의 모터 하우징을 **합성 RGB-D로 인식(S1)** → **MuJoCo 팔이 접촉으로 집기(S2)** → "
                "**측정한 실패율·시간을 공장 라인에 투입(S3)**. 각 단계 eval은 설정 동결 후 1회만 실행했다.")
    s1 = _latest(ROOT / "results" / "v5" / "s1_bin", "eval-final_*")
    s2 = _latest(MUJO / "results" / "housing_pick", "eval_*")
    s3 = _latest(ROOT / "results" / "v5" / "s3_line", "confirm_*")
    for d, fn, name in ((s1, _s1, "S1"), (s2, _s2, "S2"), (s3, _s3, "S3")):
        if d is None:
            st.warning(f"{name} eval 결과 없음")
        else:
            fn(d)
        st.divider()
    st.caption("보고서: docs/v5/final_report.md · 한계: docs/v5/limitations.md")

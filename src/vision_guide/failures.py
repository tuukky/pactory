"""실패 사유 코드.

v2 손실 분해처럼 모든 실패를 **처음 실패한 단계 하나**에만 귀속한다 (한 시도에 코드 1개).
"기타(OTHER)"가 5%를 넘으면 범주를 다시 설계한다 (평가 세트를 보기 전에만).

파지 선택 동률 규칙: 대칭 후보 × 파지 목록이 모두 실패하면 작업영역 밖(UNREACHABLE)과
접근 기울기 초과(APPROACH_LIMIT) 중 많은 쪽으로 귀속한다. 같으면 UNREACHABLE
(작업영역 검사가 먼저 수행되므로 앞 단계 우선 원칙과 같다).
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import pandas as pd


class FailureCode(str, Enum):
    OK = "OK"
    NO_POINTS = "NO_POINTS"            # ROI 안 유효 Depth 점 없음
    NO_CLUSTER = "NO_CLUSTER"          # 평면 제거 후 크기 조건을 통과한 클러스터 없음
    FEATURE_FAIL = "FEATURE_FAIL"      # FPFH+RANSAC 초기 정렬 실패 (점·대응 부족)
    LOW_FITNESS = "LOW_FITNESS"        # ICP fitness 하한 미달
    HIGH_RMSE = "HIGH_RMSE"            # ICP inlier RMSE 상한 초과
    LOW_VISIBILITY = "LOW_VISIBILITY"  # 가시 표면 비율 하한 미달
    NO_GRASP = "NO_GRASP"              # 파지 후보 없음 (자세 없음·파지 목록 비어 있음)
    UNREACHABLE = "UNREACHABLE"        # TCP 목표가 작업영역 밖
    APPROACH_LIMIT = "APPROACH_LIMIT"  # 접근축 기울기 한계 초과
    NO_DETECTION = "NO_DETECTION"      # v4.2: 대상 클래스 세그멘테이션 마스크 없음
    DEPTH_INCONSISTENT = "DEPTH_INCONSISTENT"  # v4.1: 추정 자세 렌더 depth가 관측보다 앞 (자유공간 위반)
    COLOR_MISMATCH = "COLOR_MISMATCH"  # v4.1: 가시 모델 색과 관측 색 불일치
    OTHER = "OTHER"                    # 예외 등 위 범주에 없는 실패


STAGE_OF: dict[FailureCode, str] = {
    FailureCode.OK: "ok",
    FailureCode.NO_POINTS: "preprocess",
    FailureCode.NO_CLUSTER: "segment",
    FailureCode.FEATURE_FAIL: "global_registration",
    FailureCode.LOW_FITNESS: "confidence",
    FailureCode.HIGH_RMSE: "confidence",
    FailureCode.LOW_VISIBILITY: "confidence",
    FailureCode.NO_GRASP: "guidance",
    FailureCode.UNREACHABLE: "guidance",
    FailureCode.APPROACH_LIMIT: "guidance",
    FailureCode.NO_DETECTION: "segment",
    FailureCode.DEPTH_INCONSISTENT: "confidence",
    FailureCode.COLOR_MISMATCH: "confidence",
    FailureCode.OTHER: "other",
}


def majority_grasp_failure(n_unreachable: int, n_approach: int) -> FailureCode:
    """모든 파지 후보 실패 시 귀속 (동률이면 UNREACHABLE)."""
    if n_unreachable == 0 and n_approach == 0:
        return FailureCode.NO_GRASP
    return FailureCode.UNREACHABLE if n_unreachable >= n_approach else FailureCode.APPROACH_LIMIT


def failure_breakdown(codes: Iterable[FailureCode | str]) -> pd.DataFrame:
    """코드별 건수·비율 (모든 코드 행 포함, 0건도 표시)."""
    vals = [FailureCode(c).value for c in codes]
    n = len(vals)
    rows = [{"code": c.value, "stage": STAGE_OF[c], "n": vals.count(c.value),
             "share": (vals.count(c.value) / n) if n else 0.0} for c in FailureCode]
    return pd.DataFrame(rows)


def other_failure_share(df: pd.DataFrame) -> float:
    """실패 건수(OK 제외) 대비 OTHER 비율. df는 failure_breakdown 결과. 실패 0건이면 0."""
    n_fail = float(df.loc[df["code"] != FailureCode.OK.value, "n"].sum())
    n_other = float(df.loc[df["code"] == FailureCode.OTHER.value, "n"].sum())
    return n_other / n_fail if n_fail > 0 else 0.0


def other_share_ok(df: pd.DataFrame, limit: float = 0.05, basis: str = "all") -> bool:
    """OTHER 비율이 limit 이하인가. basis="all": 전체 시도 대비(기존 판정), "failures": 실패 건수 대비."""
    if basis == "failures":
        return other_failure_share(df) <= limit
    if basis != "all":
        raise ValueError(f"basis must be 'all' or 'failures': {basis!r}")
    return float(df.loc[df["code"] == FailureCode.OTHER.value, "share"].sum()) <= limit

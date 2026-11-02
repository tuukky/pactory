"""공정 그래프: 서브라인 경로, 키트 합류, 조립 이후 검사·재검사 재순환, 출하.

서브라인 내부 이동은 컨베이어(이동 0, 유한 버퍼·막힘), 구역 사이 이동은 운반(AGV 또는
instant)이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Target:
    kind: str  # "station" | "kit" | "dock" | "ship" | "scrap"
    station_id: str | None = None
    cls: str = "general"  # 입력 버퍼 슬롯 요청 종류: general | recheck
    transport: bool = False  # True = AGV 운반 구간, False = 컨베이어/종결

    @property
    def node(self) -> str:
        if self.kind == "station":
            return f"{self.station_id}_IN"
        return {"kit": "KIT", "dock": "DOCK"}[self.kind]

    @property
    def key(self) -> tuple[str, str | None, str]:
        return (self.kind, self.station_id, self.cls)


SHIP = Target("ship")
SCRAP = Target("scrap")


class ProcessGraph:
    def __init__(self, cfg: Mapping):
        r = cfg["routing"]
        self.parts: dict[str, Mapping] = dict(r["parts"])
        self.assembly: str | None = r.get("assembly")
        insp = r.get("inspect") or {}
        self.inspect: str | None = insp.get("station") if self.assembly else None
        self.rework: str | None = insp.get("rework") if self.assembly else None
        self.pass_to: str | None = insp.get("pass_to") if self.assembly else None
        self.max_rework = int(insp.get("max_rework", 0)) if self.assembly else 0
        self.ship_to: str = r.get("ship_to") or "SHIP"
        self.stations = cfg["stations"]
        self.position: dict[str, tuple[str, int]] = {}
        for p, pc in self.parts.items():
            for i, sid in enumerate(pc["path"]):
                self.position[sid] = (p, i)

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "ProcessGraph":
        return cls(cfg)

    def source_node(self, part: str) -> str:
        return self.parts[part]["from"]

    def first_target(self, part: str) -> Target:
        path = self.parts[part]["path"]
        if path:
            return Target("station", path[0], transport=True)
        return Target("kit", transport=True)

    def next_target(self, sid: str, *, passed: bool | None = None, recheck: bool = False) -> Target:
        """sid 가공 종료 후 목적지. 검사 공정은 passed(합격 여부)와 recheck(재검사 여부)를 받는다."""
        if sid in self.position:
            p, i = self.position[sid]
            path = self.parts[p]["path"]
            if i + 1 < len(path):
                return Target("station", path[i + 1])
            return Target("kit", transport=True) if self.parts[p]["to"] == "KIT" else SHIP
        if sid == self.assembly:
            return Target("station", self.inspect, transport=True)
        if sid == self.inspect:
            if passed:
                return Target("station", self.pass_to, transport=True)
            if recheck or self.max_rework < 1:
                return SCRAP
            return Target("station", self.rework, transport=True)
        if sid == self.rework:
            return Target("station", self.inspect, cls="recheck", transport=True)
        if sid == self.pass_to:
            return Target("dock", transport=True) if self.ship_to == "DOCK" else SHIP
        raise KeyError(f"경로에 없는 스테이션: {sid}")

    def transport_out_stations(self) -> list[str]:
        """출력 버퍼(AGV 픽업)가 필요한 스테이션."""
        out = []
        for sid in self.stations:
            if sid in self.position:
                p, i = self.position[sid]
                if i == len(self.parts[p]["path"]) - 1 and self.parts[p]["to"] == "KIT":
                    out.append(sid)
            elif sid in (self.assembly, self.inspect, self.rework):
                out.append(sid)
            elif sid == self.pass_to and self.ship_to == "DOCK":
                out.append(sid)
        return out

    def operator_mode(self, sid: str) -> str:
        return self.stations[sid].get("operator", "none")

    def dispatch_rule(self, sid: str) -> str:
        return self.stations[sid].get("dispatch", "FIFO")

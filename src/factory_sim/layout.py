"""설명용 2D 레이아웃: 노드 좌표(m)와 거리 = 맨해튼 거리 × 우회계수."""

from __future__ import annotations

from typing import Mapping


class Layout:
    def __init__(self, nodes: Mapping[str, tuple[float, float]], detour: float):
        self.nodes = {str(k): (float(v[0]), float(v[1])) for k, v in nodes.items()}
        self.detour = float(detour)
        self._cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "Layout":
        lay = cfg.get("layout") or {}
        return cls(lay.get("nodes") or {}, lay.get("detour_factor", 1.0))

    def distance(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b)
        d = self._cache.get(key)
        if d is None:
            (x1, y1), (x2, y2) = self.nodes[a], self.nodes[b]
            d = (abs(x1 - x2) + abs(y1 - y2)) * self.detour
            self._cache[key] = self._cache[(b, a)] = d
        return d

    def travel_time(self, a: str, b: str, speed_mps: float) -> float:
        d = self.distance(a, b)
        return 0.0 if d == 0 else d / float(speed_mps)

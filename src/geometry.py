"""Bounding-box dataclass and geometry helpers (IoU, NMS)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Detection:
    """One detected advertisement region (pixel coordinates, top-left origin)."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0
    source: str = "heuristic"          # which backend produced it
    label: str = "advertisement"
    # filled in by the sizing step:
    width_cm: float | None = None
    height_cm: float | None = None
    area_cm2: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width_px(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height_px(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area_px(self) -> int:
        return self.width_px * self.height_px

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        known = {f for f in Detection.__dataclass_fields__}  # type: ignore[attr-defined]
        return Detection(**{k: v for k, v in d.items() if k in known})


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.area_px + b.area_px - inter
    return inter / union if union else 0.0


def nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    """Greedy non-max suppression. Keeps the higher-scoring box on overlap."""
    kept: list[Detection] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(iou(det, k) < iou_thr for k in kept):
            kept.append(det)
    return kept

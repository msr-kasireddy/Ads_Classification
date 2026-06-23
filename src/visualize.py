"""Draw advertisement boxes + cm^2 labels onto a page image."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .geometry import Detection


def draw_detections(
    image_bgr: np.ndarray, dets: list[Detection], cfg: Config
) -> np.ndarray:
    out = image_bgr.copy()
    o = cfg.get("output", default={}) or {}
    color = tuple(int(c) for c in o.get("box_color", [0, 0, 255]))
    thickness = int(o.get("box_thickness", 4))
    font_scale = float(o.get("font_scale", 0.9))
    draw_label = bool(o.get("draw_area_label", True))

    for i, d in enumerate(dets, start=1):
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        if not draw_label:
            continue
        if d.area_cm2 is not None:
            text = f"#{i} {d.area_cm2:.0f} cm2 ({d.width_cm:.1f}x{d.height_cm:.1f})"
        else:
            text = f"#{i} ad"
        _label(out, text, d.x1, d.y1, color, font_scale, thickness)
    return out


def _label(img, text, x, y, color, font_scale, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, thickness // 2)
    (tw, tht), base = cv2.getTextSize(text, font, font_scale, th)
    y_top = max(0, y - tht - base - 4)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + tht + base + 4), color, -1)
    cv2.putText(
        img, text, (x + 3, y_top + tht + 2), font, font_scale,
        (255, 255, 255), th, cv2.LINE_AA,
    )

"""Classical-CV advertisement detector — zero downloads, fully offline.

Idea: most newspaper DISPLAY ads are rectangles framed by ruled lines (a box
border) and/or contain dense graphics / large type that breaks the body-text
column grid. We:

  1. binarise the page to an ink mask,
  2. extract long horizontal + vertical ruled lines (morphology),
  3. close them into closed rectangles and find contours,
  4. keep rectangle-ish contours whose interior is sufficiently "inky"
     (graphics / big fonts) rather than empty.

This is intentionally simple and explainable. It gives a usable Day-1 baseline
with no model weights; the dashboard correction loop + YOLOv8 fine-tune is what
pushes accuracy up on real Telugu pages.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..geometry import Detection, nms


class HeuristicAdDetector:
    name = "heuristic"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        h = cfg.get("heuristic", default={}) or {}
        self.line_len_frac = float(h.get("line_len_frac", 0.020))
        self.close_frac = float(h.get("close_frac", 0.012))
        self.min_aspect = float(h.get("min_box_aspect", 0.10))
        self.max_aspect = float(h.get("max_box_aspect", 10.0))
        self.min_fill = float(h.get("min_fill", 0.04))
        self.nms_iou = float(cfg.get("detection", "nms_iou", default=0.40))

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        # Binary ink mask (white = ink). Otsu adapts to scan brightness.
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- extract ruled border lines -----------------------------------
        h_len = max(15, int(W * self.line_len_frac))
        v_len = max(15, int(H * self.line_len_frac))
        horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_k)
        vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vert_k)
        lines = cv2.bitwise_or(horiz, vert)

        # close gaps so a box border becomes a solid loop
        c = max(3, int(min(W, H) * self.close_frac))
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (c, c))
        closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_k)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        page_area = float(W * H)
        min_side = 0.03 * min(W, H)   # ignore tiny boxes
        dets: list[Detection] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_side or h < min_side:
                continue
            aspect = w / float(h)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue

            # Verify the bbox is an actual ruled BOX: its four edges must be
            # substantially covered by ruled lines. This rejects blobs formed
            # by merging unrelated text/graphics.
            perim = self._perimeter_coverage(lines, x, y, w, h)
            if perim < 0.55:
                continue

            # how inky is the interior (graphics / large type)?
            interior = ink[y : y + h, x : x + w]
            fill = float(interior.mean()) / 255.0
            if fill < self.min_fill:
                continue
            rect_fill = perim
            score = 0.4 + 0.4 * min(1.0, perim) + 0.2 * min(1.0, fill * 4)
            d = Detection(
                x1=x, y1=y, x2=x + w, y2=y + h,
                score=round(score, 3), source=self.name,
            )
            d.extra = {
                "rect_fill": round(rect_fill, 3),
                "ink_fill": round(fill, 3),
                "area_frac": round((w * h) / page_area, 4),
            }
            dets.append(d)

        return nms(dets, self.nms_iou)

    @staticmethod
    def _perimeter_coverage(lines: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Fraction of the bbox perimeter that lies on ruled lines, averaged
        over the 4 edges. A real boxed ad scores high; a merged text blob low."""
        # band must be >= the close-kernel offset between the contour bbox and
        # the true ruled border, else the edge falls just outside the band.
        t = max(8, int(0.015 * max(lines.shape)))  # edge band thickness
        H, W = lines.shape

        def band(y0, y1, x0, x1) -> float:
            y0, y1 = max(0, y0), min(H, y1)
            x0, x1 = max(0, x0), min(W, x1)
            if y1 <= y0 or x1 <= x0:
                return 0.0
            strip = lines[y0:y1, x0:x1]
            if strip.size == 0:
                return 0.0
            # along the long axis, an edge pixel is "covered" if any ink in band
            axis = 0 if (y1 - y0) <= (x1 - x0) else 1
            covered = (strip > 0).max(axis=axis)
            return float(covered.mean())

        top = band(y - t, y + t, x, x + w)
        bot = band(y + h - t, y + h + t, x, x + w)
        left = band(y, y + h, x - t, x + t)
        right = band(y, y + h, x + w - t, x + w + t)
        return (top + bot + left + right) / 4.0

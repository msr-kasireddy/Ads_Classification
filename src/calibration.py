"""Pixel -> centimetre calibration.

The cm^2 sizes are only meaningful if we know how many centimetres a pixel
represents. Two strategies are supported:

* ``auto_content`` (default, recommended): detect the printed-content bounding
  box on the page and map its WIDTH to the publication's known print-area width
  (e.g. 33 cm for Sakshi/Eenadu). This is robust to unknown DPI and to white
  scanner margins around the page.

* ``dpi``: trust a fixed scan DPI, so ``cm = px * 2.54 / dpi``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Calibration:
    cm_per_px_x: float
    cm_per_px_y: float
    # bounding box of detected printed content (x1, y1, x2, y2) in px
    content_box: tuple[int, int, int, int]
    mode: str

    def cm2(self, width_px: float, height_px: float) -> tuple[float, float, float]:
        w = width_px * self.cm_per_px_x
        h = height_px * self.cm_per_px_y
        return w, h, w * h


def find_content_box(
    gray: np.ndarray, ink_threshold: int = 200, min_ink_fraction: float = 0.01
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) of the printed area, ignoring white margins.

    A row/column is considered part of the page content if the fraction of
    'ink' pixels (darker than ``ink_threshold``) exceeds ``min_ink_fraction``.
    """
    h, w = gray.shape[:2]
    ink = (gray < ink_threshold).astype(np.uint8)

    col_frac = ink.sum(axis=0) / float(h)
    row_frac = ink.sum(axis=1) / float(w)

    cols = np.where(col_frac > min_ink_fraction)[0]
    rows = np.where(row_frac > min_ink_fraction)[0]

    if cols.size == 0 or rows.size == 0:
        return 0, 0, w, h  # fall back to full image
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def calibrate(
    image_bgr: np.ndarray,
    page_width_cm: float,
    page_height_cm: float,
    *,
    mode: str = "auto_content",
    dpi: float = 150.0,
    ink_threshold: int = 200,
    min_ink_fraction: float = 0.01,
) -> Calibration:
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )
    box = find_content_box(gray, ink_threshold, min_ink_fraction)
    x1, y1, x2, y2 = box

    if mode == "dpi":
        cm_per_px = 2.54 / float(dpi)
        return Calibration(cm_per_px, cm_per_px, box, "dpi")

    # auto_content
    content_w = max(1, x2 - x1)
    content_h = max(1, y2 - y1)
    cm_per_px_x = page_width_cm / content_w
    # Prefer a single uniform scale (square pixels) anchored on width, which is
    # the most reliable dimension; only fall back to height if width looks off.
    cm_per_px_y = cm_per_px_x
    # sanity: if implied page height is wildly different from the known height,
    # the scan is non-uniform — calibrate the two axes independently.
    implied_h_cm = content_h * cm_per_px_x
    if page_height_cm > 0 and abs(implied_h_cm - page_height_cm) / page_height_cm > 0.15:
        cm_per_px_y = page_height_cm / content_h
    return Calibration(cm_per_px_x, cm_per_px_y, box, "auto_content")

"""Attach real-world cm^2 sizes to detections and apply size-based filtering."""
from __future__ import annotations

from .calibration import Calibration
from .config import Config
from .geometry import Detection


def apply_sizes(
    dets: list[Detection],
    calib: Calibration,
    cfg: Config,
    image_wh: tuple[int, int],
) -> list[Detection]:
    """Fill width_cm/height_cm/area_cm2 on each detection and drop ads that are
    too small to be real (noise) per ``detection.min_ad_area_cm2``."""
    min_area = float(cfg.get("detection", "min_ad_area_cm2", default=15.0))
    max_frac = float(cfg.get("detection", "max_ad_area_frac", default=0.95))
    W, H = image_wh
    page_px = float(W * H)

    out: list[Detection] = []
    for d in dets:
        w_cm, h_cm, area = calib.cm2(d.width_px, d.height_px)
        d.width_cm = round(w_cm, 2)
        d.height_cm = round(h_cm, 2)
        d.area_cm2 = round(area, 2)
        if area < min_area:
            continue
        if page_px and (d.area_px / page_px) > max_frac:
            d.extra = {**d.extra, "full_page": True}
        out.append(d)
    return out


def total_ad_area_cm2(dets: list[Detection]) -> float:
    return round(sum(d.area_cm2 or 0.0 for d in dets), 2)

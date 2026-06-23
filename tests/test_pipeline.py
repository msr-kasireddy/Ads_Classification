"""Smoke tests for the ad-detection pipeline (heuristic backend, no downloads)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_sample import make_page  # noqa: E402
from src.calibration import calibrate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.geometry import Detection, iou, nms  # noqa: E402
from src.pipeline import AdsPipeline  # noqa: E402


def test_iou_and_nms():
    a = Detection(0, 0, 10, 10, score=0.9)
    b = Detection(0, 0, 10, 10, score=0.5)
    assert iou(a, b) == 1.0
    assert len(nms([a, b], 0.5)) == 1


def test_calibration_scales_to_page_width():
    img = make_page(1650, 2600)
    calib = calibrate(img, 33.0, 52.0, mode="auto_content")
    # a full-page-width span should map to ~33 cm
    span_cm = (calib.content_box[2] - calib.content_box[0]) * calib.cm_per_px_x
    assert 30.0 < span_cm < 36.0


def test_pipeline_detects_synthetic_ads(tmp_path):
    img = make_page()
    p = tmp_path / "sample_sakshi.png"
    import cv2

    cv2.imwrite(str(p), img)

    pipe = AdsPipeline(load_config())
    res = pipe.process_image(p, save=False)

    assert res.num_ads >= 3
    assert res.total_ad_area_cm2 > 0
    for d in res.detections:
        assert d["area_cm2"] is not None and d["area_cm2"] > 0


if __name__ == "__main__":
    test_iou_and_nms()
    test_calibration_scales_to_page_width()
    import tempfile

    test_pipeline_detects_synthetic_ads(Path(tempfile.mkdtemp()))
    print("all tests passed")

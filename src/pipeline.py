"""End-to-end pipeline: image path -> detections + cm^2 sizes + outputs."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .calibration import Calibration, calibrate
from .config import Config, load_config
from .detectors import build_detector
from .geometry import Detection
from .sizing import apply_sizes, total_ad_area_cm2
from .visualize import draw_detections

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class PageResult:
    image_path: str
    width_px: int
    height_px: int
    publication: str
    page_width_cm: float
    page_height_cm: float
    calibration_mode: str
    cm_per_px_x: float
    cm_per_px_y: float
    num_ads: int
    total_ad_area_cm2: float
    page_area_cm2: float
    ad_coverage_pct: float
    detections: list[dict] = field(default_factory=list)
    annotated_path: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def guess_publication(path: Path) -> str:
    s = str(path).lower()
    for pub in ("sakshi", "eenadu", "andhrajyothi", "namasthe", "vaartha"):
        if pub in s:
            return pub
    return path.parent.name or "default"


class AdsPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.detector = build_detector(self.cfg)

    # -- single image -----------------------------------------------------
    def process_image(
        self, image_path: str | Path, *, save: bool = True, out_dir: Path | None = None
    ) -> PageResult:
        image_path = Path(image_path)
        t0 = time.time()
        img = _imread_unicode(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        H, W = img.shape[:2]

        publication = guess_publication(image_path)
        pw_cm, ph_cm = self.cfg.page_size_cm(publication)

        cal_cfg = self.cfg.get("calibration", default={}) or {}
        calib: Calibration = calibrate(
            img, pw_cm, ph_cm,
            mode=cal_cfg.get("mode", "auto_content"),
            dpi=float(cal_cfg.get("dpi", 150)),
            ink_threshold=int(cal_cfg.get("ink_threshold", 200)),
            min_ink_fraction=float(cal_cfg.get("min_ink_fraction", 0.01)),
        )

        dets: list[Detection] = self.detector.detect(img)
        dets = apply_sizes(dets, calib, self.cfg, (W, H))
        dets.sort(key=lambda d: d.area_cm2 or 0.0, reverse=True)

        page_area_cm2 = round(pw_cm * ph_cm, 2)
        total_area = total_ad_area_cm2(dets)
        coverage = round(100.0 * total_area / page_area_cm2, 2) if page_area_cm2 else 0.0

        annotated_path = None
        if save:
            out_dir = Path(out_dir or (self.cfg.get("output", "dir", default="data/output")))
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated = draw_detections(img, dets, self.cfg)
            annotated_path = str(out_dir / f"{image_path.stem}_ads.jpg")
            _imwrite_unicode(annotated_path, annotated)

        result = PageResult(
            image_path=str(image_path),
            width_px=W, height_px=H,
            publication=publication,
            page_width_cm=pw_cm, page_height_cm=ph_cm,
            calibration_mode=calib.mode,
            cm_per_px_x=round(calib.cm_per_px_x, 6),
            cm_per_px_y=round(calib.cm_per_px_y, 6),
            num_ads=len(dets),
            total_ad_area_cm2=total_area,
            page_area_cm2=page_area_cm2,
            ad_coverage_pct=coverage,
            detections=[d.to_dict() for d in dets],
            annotated_path=annotated_path,
            elapsed_s=round(time.time() - t0, 3),
        )

        if save and out_dir is not None:
            with open(out_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
        return result

    # -- folder -----------------------------------------------------------
    def process_folder(
        self, folder: str | Path, *, recursive: bool = True
    ) -> list[PageResult]:
        folder = Path(folder)
        out_dir = Path(self.cfg.get("output", "dir", default="data/output"))
        results: list[PageResult] = []
        for img_path in iter_images(folder, recursive):
            try:
                res = self.process_image(img_path, save=True, out_dir=out_dir)
                results.append(res)
                print(
                    f"[ok] {img_path.name}: {res.num_ads} ads, "
                    f"{res.total_ad_area_cm2:.0f} cm2 ({res.ad_coverage_pct:.1f}% of page)"
                )
            except Exception as exc:  # keep going through the batch
                print(f"[err] {img_path}: {exc}")
        _write_summary(out_dir, results)
        return results


def iter_images(folder: Path, recursive: bool = True) -> Iterable[Path]:
    globber = folder.rglob("*") if recursive else folder.glob("*")
    for p in sorted(globber):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _write_summary(out_dir: Path, results: list[PageResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image": Path(r.image_path).name,
            "publication": r.publication,
            "num_ads": r.num_ads,
            "total_ad_area_cm2": r.total_ad_area_cm2,
            "page_area_cm2": r.page_area_cm2,
            "ad_coverage_pct": r.ad_coverage_pct,
            "annotated": r.annotated_path,
        }
        for r in results
    ]
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\nSummary written -> {out_dir / 'summary.json'} ({len(rows)} pages)")


# -- unicode-safe image IO (handles non-ASCII Telugu paths) ----------------
def _imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)

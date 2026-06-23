#!/usr/bin/env python3
"""Batch-process a folder of newspaper page images.

Usage:
    python scripts/run_batch.py /path/to/scraped_images
    python scripts/run_batch.py /path/to/scraped_images --backend ensemble
    python scripts/run_batch.py one_page.jpg          # single file

Outputs (under config `output.dir`, default data/output/):
    <name>_ads.jpg   annotated page with boxes + cm^2 labels
    <name>.json      per-page detections + sizes
    summary.json     one row per page
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.pipeline import AdsPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect newspaper ads + sizes (cm^2).")
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--backend", help="override detection.backend "
                    "(heuristic|doclayout|yolov8|ensemble)")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--no-recursive", action="store_true",
                    help="do not descend into sub-folders")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg.raw.setdefault("detection", {})["backend"] = args.backend

    pipe = AdsPipeline(cfg)
    target = Path(args.path)
    if target.is_dir():
        pipe.process_folder(target, recursive=not args.no_recursive)
    else:
        res = pipe.process_image(target)
        print(f"{res.num_ads} ads, {res.total_ad_area_cm2:.0f} cm2, "
              f"annotated -> {res.annotated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

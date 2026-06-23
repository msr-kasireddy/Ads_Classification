"""Streamlit dashboard for newspaper ad detection.

Run:
    pip install -r requirements.txt
    streamlit run app/dashboard.py

Features
  * Browse every page in a folder (your scraped_images/).
  * See detected advertisements boxed, with per-ad cm^2 sizes and page coverage.
  * CORRECT the boxes (add / delete / adjust) in an editable table, save them.
  * Export all corrected pages to a YOLO dataset for local fine-tuning.

All local. No internet, no API keys, no cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.geometry import Detection  # noqa: E402
from src.pipeline import AdsPipeline, iter_images  # noqa: E402
from src.visualize import draw_detections  # noqa: E402

CORRECTIONS_DIR = PROJECT_ROOT / "data" / "corrections"
st.set_page_config(page_title="Newspaper Ads Detection", layout="wide")


@st.cache_resource
def get_pipeline(backend: str, config_path: str):
    cfg = load_config(config_path or None)
    if backend:
        cfg.raw.setdefault("detection", {})["backend"] = backend
    return AdsPipeline(cfg), cfg


@st.cache_data(show_spinner=False)
def list_pages(folder: str) -> list[str]:
    return [str(p) for p in iter_images(Path(folder), recursive=True)]


def corrections_path(image_path: str) -> Path:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return CORRECTIONS_DIR / f"{Path(image_path).stem}.json"


def main() -> None:
    st.title("📰 Newspaper Advertisement Detection")
    st.caption("Detect ads, measure their size in cm², draw boxes — fully local.")

    with st.sidebar:
        st.header("Settings")
        default_folder = str(PROJECT_ROOT / "data" / "images")
        folder = st.text_input("Image folder", value=default_folder,
                               help="Point this at your scraped_images/ folder.")
        backend = st.selectbox(
            "Detector backend",
            ["heuristic", "ensemble", "doclayout", "yolov8"],
            help="heuristic = no downloads. yolov8 = your fine-tuned model.",
        )
        config_path = st.text_input("config.yaml", value=str(PROJECT_ROOT / "config.yaml"))
        st.divider()
        if st.button("Export corrections → YOLO dataset", use_container_width=True):
            _export(folder)

    if not folder or not Path(folder).is_dir():
        st.info("Enter a valid image folder in the sidebar to begin.")
        return

    pages = list_pages(folder)
    if not pages:
        st.warning(f"No images found under {folder}")
        return

    pipeline, cfg = get_pipeline(backend, config_path)

    idx = st.sidebar.number_input("Page", 0, len(pages) - 1, 0, 1)
    st.sidebar.write(f"{idx + 1} / {len(pages)} pages")
    image_path = pages[int(idx)]

    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Could not read {image_path}")
        return

    # detection (cached corrections take priority for the editable table)
    result = pipeline.process_image(image_path, save=False)
    dets = [Detection.from_dict(d) for d in result.detections]

    saved = corrections_path(image_path)
    using_saved = False
    if saved.exists():
        rec = json.loads(saved.read_text(encoding="utf-8"))
        if rec.get("boxes"):
            dets = _boxes_to_dets(rec["boxes"], img, cfg)
            using_saved = True

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader(Path(image_path).name)
        annotated = draw_detections(img, dets, cfg)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Ads detected", len(dets))
        total = round(sum(d.area_cm2 or 0 for d in dets), 1)
        c2.metric("Total ad area", f"{total:.0f} cm²")
        page_area = result.page_area_cm2 or 1
        c3.metric("Page coverage", f"{100*total/page_area:.1f}%")
        st.caption(
            f"Publication: **{result.publication}**  ·  "
            f"page {result.page_width_cm:.0f}×{result.page_height_cm:.0f} cm  ·  "
            f"calib: {result.calibration_mode} "
            f"({result.cm_per_px_x:.4f} cm/px)"
            + ("  ·  ✏️ showing saved corrections" if using_saved else "")
        )

        st.markdown("**Correct the boxes** (edit x/y, delete rows, or add new):")
        df = pd.DataFrame(
            [{"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
              "area_cm2": d.area_cm2} for d in dets]
        )
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True,
            key=f"editor_{idx}",
            column_config={"area_cm2": st.column_config.NumberColumn(
                "area_cm²", disabled=True)},
        )

        b1, b2 = st.columns(2)
        if b1.button("💾 Save corrections", use_container_width=True):
            boxes = _df_to_boxes(edited)
            corrections_path(image_path).write_text(
                json.dumps({"image_path": image_path, "boxes": boxes},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            st.success(f"Saved {len(boxes)} boxes. Reloading…")
            st.rerun()
        if b2.button("↩︎ Reset to auto-detect", use_container_width=True):
            if saved.exists():
                saved.unlink()
            st.rerun()


def _df_to_boxes(df: pd.DataFrame) -> list[list[int]]:
    boxes = []
    for _, r in df.iterrows():
        try:
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        except (ValueError, TypeError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _boxes_to_dets(boxes, img, cfg) -> list[Detection]:
    from src.calibration import calibrate
    from src.sizing import apply_sizes

    H, W = img.shape[:2]
    cal_cfg = cfg.get("calibration", default={}) or {}
    pw, ph = cfg.page_size_cm("default")
    calib = calibrate(img, pw, ph, mode=cal_cfg.get("mode", "auto_content"),
                      dpi=float(cal_cfg.get("dpi", 150)))
    dets = [Detection(x1=b[0], y1=b[1], x2=b[2], y2=b[3], source="manual")
            for b in boxes]
    return apply_sizes(dets, calib, cfg, (W, H))


def _export(folder: str) -> None:
    from src.dataset_export import export_yolo_dataset

    try:
        out = PROJECT_ROOT / "data" / "yolo_dataset"
        yaml_path = export_yolo_dataset(CORRECTIONS_DIR, out)
        st.sidebar.success(f"Dataset → {yaml_path.parent}")
    except Exception as exc:
        st.sidebar.error(str(exc))


if __name__ == "__main__":
    main()

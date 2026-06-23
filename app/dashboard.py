"""Streamlit dashboard for newspaper ad detection — end-user friendly.

Launch by DOUBLE-CLICKING the launcher (run_dashboard.command on macOS,
run_dashboard.bat on Windows). No command line needed.

What you can do here
  * Browse every page in your scraped_images/ folder.
  * See what the current model detects (red boxes) with each ad's size in cm².
  * DRAW missing ads with your mouse, MOVE / RESIZE / DELETE wrong ones.
  * Save your corrections (builds your training set).
  * Train your own model on the corrected pages — accuracy improves over time.

Everything runs locally. No internet, no accounts, no cost.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import calibrate  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.geometry import Detection  # noqa: E402
from src.pipeline import AdsPipeline, iter_images  # noqa: E402

try:
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except Exception:  # pragma: no cover
    HAVE_CANVAS = False

CORRECTIONS_DIR = PROJECT_ROOT / "data" / "corrections"
DISPLAY_W = 1000  # canvas width in px; pages are downscaled to this
st.set_page_config(page_title="Newspaper Ads — Detect & Train", layout="wide")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_pipeline(backend: str, config_path: str):
    cfg = load_config(config_path or None)
    if backend:
        cfg.raw.setdefault("detection", {})["backend"] = backend
    return AdsPipeline(cfg), cfg


@st.cache_data(show_spinner=False)
def list_pages(folder: str) -> list[str]:
    return [str(p) for p in iter_images(Path(folder), recursive=True)]


@st.cache_data(show_spinner="Detecting ads…")
def detect_boxes(image_path: str, backend: str, config_path: str) -> list[list[int]]:
    pipe, _ = get_pipeline(backend, config_path)
    res = pipe.process_image(image_path, save=False)
    return [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in res.detections]


def corrections_path(image_path: str) -> Path:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return CORRECTIONS_DIR / f"{Path(image_path).stem}.json"


def load_saved_boxes(image_path: str) -> list[list[int]] | None:
    p = corrections_path(image_path)
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec.get("boxes")
    return None


def boxes_to_initial_drawing(boxes: list[list[int]], scale: float) -> dict:
    """Pre-load existing boxes (original px) onto the canvas (display px)."""
    objects = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        objects.append({
            "type": "rect",
            "left": x1 * scale, "top": y1 * scale,
            "width": (x2 - x1) * scale, "height": (y2 - y1) * scale,
            "fill": "rgba(255, 0, 0, 0.10)",
            "stroke": "#FF0000", "strokeWidth": 2,
        })
    return {"version": "4.4.0", "objects": objects}


def canvas_to_boxes(canvas_json: dict | None, scale: float) -> list[list[int]]:
    """Read rectangles back from the canvas → original-image pixel boxes."""
    if not canvas_json:
        return []
    out = []
    for o in canvas_json.get("objects", []):
        if o.get("type") != "rect":
            continue
        w = o["width"] * o.get("scaleX", 1)
        h = o["height"] * o.get("scaleY", 1)
        x1, y1 = o["left"], o["top"]
        x2, y2 = x1 + w, y1 + h
        box = [int(round(v / scale)) for v in (x1, y1, x2, y2)]
        if box[2] - box[0] > 3 and box[3] - box[1] > 3:
            out.append(box)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📰 Newspaper Advertisement Detection & Training")

    with st.sidebar:
        st.header("① Your pages")
        folder = st.text_input(
            "Folder of newspaper images",
            value=str(PROJECT_ROOT / "data" / "images"),
            help="Point this at your scraped_images/ folder.",
        )
        backend = st.selectbox(
            "Detector",
            ["heuristic", "yolov8", "ensemble", "doclayout"],
            help="heuristic = built-in candidate finder (no setup). "
                 "yolov8 = your trained model (most accurate).",
        )
        config_path = str(PROJECT_ROOT / "config.yaml")

    if not folder or not Path(folder).is_dir():
        st.info("⬅️ Enter a valid image folder in the sidebar to begin.")
        return
    pages = list_pages(folder)
    if not pages:
        st.warning(f"No images found under {folder}")
        return

    # page navigation -------------------------------------------------------
    if "pidx" not in st.session_state:
        st.session_state.pidx = 0
    with st.sidebar:
        st.header("② Browse")
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Prev", use_container_width=True):
            st.session_state.pidx = max(0, st.session_state.pidx - 1)
        if c2.button("Next ➡️", use_container_width=True):
            st.session_state.pidx = min(len(pages) - 1, st.session_state.pidx + 1)
        st.session_state.pidx = st.number_input(
            "Page #", 0, len(pages) - 1, st.session_state.pidx, 1)
        st.caption(f"{st.session_state.pidx + 1} of {len(pages)} pages")

    image_path = pages[int(st.session_state.pidx)]
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Could not read {image_path}")
        return
    H, W = img.shape[:2]
    scale = DISPLAY_W / W
    disp_h = int(H * scale)
    bg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((DISPLAY_W, disp_h))

    # boxes: saved corrections take priority, else model detections
    saved = load_saved_boxes(image_path)
    boxes = saved if saved is not None else detect_boxes(image_path, backend, config_path)
    source_label = "✏️ your saved corrections" if saved is not None else f"🤖 {backend} detections"

    # calibration (for cm² readout) ----------------------------------------
    _, cfg = get_pipeline(backend, config_path)
    pub = _publication(image_path)
    pw, ph = cfg.page_size_cm(pub)
    calib = calibrate(img, pw, ph,
                      mode=cfg.get("calibration", "mode", default="auto_content"),
                      dpi=float(cfg.get("calibration", "dpi", default=150)))

    st.subheader(f"{Path(image_path).name}  ·  {source_label}")
    mode_label = st.radio(
        "Tool", ["✏️ Draw new ad", "✋ Move / resize / delete"],
        horizontal=True,
        help="Draw: drag a box over a missed ad. "
             "Move/resize/delete: click a box to select, drag handles, or press Delete.",
    )
    draw_mode = "rect" if mode_label.startswith("✏️") else "transform"

    left, right = st.columns([3, 2], gap="large")

    with left:
        if HAVE_CANVAS:
            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.10)",
                stroke_color="#FF0000",
                stroke_width=2,
                background_image=bg,
                update_streamlit=True,
                height=disp_h,
                width=DISPLAY_W,
                drawing_mode=draw_mode,
                initial_drawing=boxes_to_initial_drawing(boxes, scale),
                display_toolbar=True,
                key=f"canvas_{st.session_state.pidx}_{'s' if saved is not None else 'd'}",
            )
            current = canvas_to_boxes(canvas.json_data if canvas else None, scale)
        else:
            st.warning("Interactive drawing needs: pip install streamlit-drawable-canvas")
            st.image(bg, use_container_width=True)
            current = boxes

    with right:
        total = 0.0
        rows = []
        for i, b in enumerate(current, 1):
            w_cm, h_cm, area = calib.cm2(b[2] - b[0], b[3] - b[1])
            total += area
            rows.append({"#": i, "w_cm": round(w_cm, 1),
                         "h_cm": round(h_cm, 1), "area_cm²": round(area, 1)})
        m1, m2, m3 = st.columns(3)
        m1.metric("Ads on page", len(current))
        m2.metric("Total ad area", f"{total:.0f} cm²")
        page_area = pw * ph
        m3.metric("Page coverage", f"{100*total/page_area:.1f}%")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=240)

        if st.button("💾 Save my corrections for this page",
                     type="primary", use_container_width=True):
            corrections_path(image_path).write_text(
                json.dumps({"image_path": image_path, "boxes": current},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Saved {len(current)} ad boxes for this page.")

        if saved is not None and st.button("↩︎ Reset this page to auto-detect",
                                           use_container_width=True):
            corrections_path(image_path).unlink(missing_ok=True)
            st.rerun()

    _training_section(folder)


def _publication(image_path: str) -> str:
    s = image_path.lower()
    for p in ("sakshi", "eenadu", "andhrajyothi", "vaartha"):
        if p in s:
            return p
    return "default"


def _training_section(folder: str) -> None:
    st.divider()
    st.header("③ Train your own model")
    n_done = len(list(CORRECTIONS_DIR.glob("*.json"))) if CORRECTIONS_DIR.exists() else 0
    st.write(f"You have corrected **{n_done}** page(s). "
             "Aim for a few hundred for high accuracy.")

    with st.expander("❓ How does it learn from my fixes? (important)"):
        st.markdown(
            "- Every box you **keep** on a saved page = *this is an ad*.\n"
            "- Every box you **delete** (and anything you leave unboxed) = "
            "*this is NOT an ad* — the model learns to stop marking it.\n"
            "- So to fix a **wrongly-marked ad**: switch to **✋ Move / resize / "
            "delete**, click the wrong red box, press **Delete**, then **💾 Save**.\n"
            "- To add a **missed ad**: switch to **✏️ Draw new ad**, drag a box, "
            "then **💾 Save**.\n\n"
            "Training uses *only your saved pages* as the truth. The more pages "
            "you correct, the more accurate your model becomes. After training, "
            "set the **Detector** (sidebar) to **yolov8** and it will mark far "
            "fewer wrong ads."
        )
    have_trainer = _module_available("ultralytics")

    c1, c2 = st.columns(2)
    with c1:
        if not have_trainer:
            if st.button("①  Set up the AI trainer (one-time ~2 GB download)",
                         use_container_width=True):
                with st.status("Installing the trainer (ultralytics)…", expanded=True) as s:
                    _pip_install("ultralytics")
                    s.update(label="Trainer installed. Reload the page.", state="complete")
        else:
            st.success("AI trainer is installed ✅")
    with c2:
        disabled = (not have_trainer) or n_done < 1
        if st.button("②  Train on my corrected pages", type="primary",
                     disabled=disabled, use_container_width=True):
            _run_training()

    st.caption("Training runs fully on your computer (GPU if available, else CPU). "
               "When it finishes, switch the Detector (sidebar) to **yolov8** to use it.")


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _pip_install(pkg: str) -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                          capture_output=True, text=True)
    st.text(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        st.error(proc.stderr[-2000:])


def _run_training() -> None:
    from src.dataset_export import export_yolo_dataset

    with st.status("Preparing dataset and training…", expanded=True) as s:
        try:
            export_yolo_dataset(CORRECTIONS_DIR, PROJECT_ROOT / "data" / "yolo_dataset")
        except Exception as exc:
            st.error(f"Could not build dataset: {exc}")
            return
        st.write("Dataset ready. Starting training (this can take a while)…")
        log = PROJECT_ROOT / "data" / "training.log"
        with open(log, "w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.train"],
                cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        st.write(f"Training started (PID {proc.pid}). Progress is written to "
                 f"`{log}`. You can keep correcting pages meanwhile.")
        s.update(label="Training launched.", state="complete")


if __name__ == "__main__":
    main()

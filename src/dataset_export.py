"""Export human-corrected detections to a YOLO detection dataset.

Workflow:
  1. Run the baseline detector over your pages.
  2. Correct the boxes in the dashboard (add / delete / adjust) -> this writes a
     corrections JSON per image into ``data/corrections/<stem>.json``.
  3. Call ``export_yolo_dataset`` (or use the dashboard button) to turn those
     corrections into a YOLO dataset you can fine-tune YOLOv8 on (src/train.py).

A correction file is simply: {"image_path": ..., "boxes": [[x1,y1,x2,y2], ...]}.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .config import PROJECT_ROOT

CLASS_NAMES = ["advertisement"]


def load_corrections(corrections_dir: Path) -> list[dict]:
    out = []
    for jf in sorted(Path(corrections_dir).glob("*.json")):
        try:
            out.append(json.loads(jf.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def export_yolo_dataset(
    corrections_dir: str | Path,
    out_dir: str | Path,
    *,
    val_split: float = 0.2,
    seed: int = 0,
) -> Path:
    corrections_dir = Path(corrections_dir)
    out_dir = Path(out_dir)
    records = [r for r in load_corrections(corrections_dir) if r.get("boxes")]
    if not records:
        raise ValueError(f"No corrections with boxes found in {corrections_dir}")

    random.Random(seed).shuffle(records)
    n_val = max(1, int(len(records) * val_split)) if len(records) > 1 else 0
    splits = {"val": records[:n_val], "train": records[n_val:]}

    for split, recs in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            _write_record(rec, out_dir, split)

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "path: {p}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: advertisement\n".format(p=out_dir.resolve()),
        encoding="utf-8",
    )
    print(f"Exported {len(records)} pages -> {out_dir} "
          f"(train={len(splits['train'])}, val={len(splits['val'])})")
    return yaml_path


def _write_record(rec: dict, out_dir: Path, split: str) -> None:
    import cv2  # local import keeps cv2 optional for callers that don't export

    src = Path(rec["image_path"])
    img = cv2.imread(str(src))
    if img is None:
        return
    H, W = img.shape[:2]
    stem = src.stem
    shutil.copy2(src, out_dir / "images" / split / src.name)

    lines = []
    for box in rec["boxes"]:
        x1, y1, x2, y2 = box[:4]
        x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
        y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out_dir / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    export_yolo_dataset(
        PROJECT_ROOT / "data" / "corrections",
        PROJECT_ROOT / "data" / "yolo_dataset",
    )

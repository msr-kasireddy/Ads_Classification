"""Fine-tune YOLOv8 on your corrected Telugu ad dataset — fully local, $0.

Prereq:  pip install ultralytics
         python -m src.dataset_export        # build data/yolo_dataset

Run:     python -m src.train                 # CPU works; GPU is much faster

The result (best.pt) is copied to models/ads_yolov8.pt. Point config.yaml at it:
    detection:
      backend: yolov8
      yolov8_weights: models/ads_yolov8.pt
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import PROJECT_ROOT


def train(
    data_yaml: str | Path | None = None,
    *,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    device: str | int | None = None,  # e.g. 0 for first GPU, "cpu" to force CPU
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Training needs: pip install ultralytics") from exc

    data_yaml = Path(data_yaml or (PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml"))
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Build it first: python -m src.dataset_export"
        )

    model = YOLO(base_model)  # downloads tiny COCO-pretrained weights once (free)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(PROJECT_ROOT / "runs"),
        name="ads_yolov8",
        exist_ok=True,
    )

    best = PROJECT_ROOT / "runs" / "ads_yolov8" / "weights" / "best.pt"
    dest = PROJECT_ROOT / "models" / "ads_yolov8.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dest)
        print(f"\nFine-tuned model -> {dest}")
        print("Set config.yaml detection.backend: yolov8 to use it.")
    return dest


if __name__ == "__main__":
    train()

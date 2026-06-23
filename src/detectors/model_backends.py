"""Deep-learning advertisement detectors (OPTIONAL backends).

These run fully locally (CPU works; GPU is faster) and are free:

* ``doclayout`` — DocLayout-YOLO, a newspaper/document-aware layout model
  (`pip install doclayout-yolo`). Auto-downloads Apache-2.0 weights from the
  free Hugging Face hub on first use. Its "figure"/"picture" regions are strong
  advertisement candidates.

* ``yolov8`` — any ultralytics YOLOv8 ``.pt`` (`pip install ultralytics`).
  Point ``detection.yolov8_weights`` at YOUR fine-tuned ad model (see
  ``src/train.py``) for the most accurate, Telugu-specific results.

Both are imported lazily; importing this module does NOT require torch unless a
model backend is actually constructed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config, PROJECT_ROOT
from ..geometry import Detection, nms


class ModelAdDetector:
    def __init__(self, cfg: Config, kind: str = "doclayout"):
        self.cfg = cfg
        self.kind = kind
        self.conf = float(cfg.get("detection", "model_conf", default=0.25))
        self.nms_iou = float(cfg.get("detection", "nms_iou", default=0.40))
        self.ad_names = {
            s.lower()
            for s in cfg.get("detection", "ad_class_names", default=[]) or []
        }
        self._model = None

    @property
    def name(self) -> str:
        return self.kind

    # -- model loading ----------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        if self.kind == "doclayout":
            self._model = self._load_doclayout()
        elif self.kind == "yolov8":
            self._model = self._load_yolov8()
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")
        return self._model

    def _load_doclayout(self):
        try:
            from doclayout_yolo import YOLOv10  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "doclayout backend needs: pip install doclayout-yolo huggingface_hub"
            ) from exc
        weights = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        return YOLOv10(weights)

    def _load_yolov8(self):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yolov8 backend needs: pip install ultralytics") from exc
        wpath = Path(self.cfg.get("detection", "yolov8_weights", default=""))
        if not wpath.is_absolute():
            wpath = PROJECT_ROOT / wpath
        if not wpath.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found: {wpath}. Train one with src/train.py "
                "or set detection.yolov8_weights in config.yaml."
            )
        return YOLO(str(wpath))

    # -- inference --------------------------------------------------------
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        model = self._load()
        imgsz = 1024 if self.kind == "doclayout" else 1280
        results = model.predict(
            image_bgr, conf=self.conf, imgsz=imgsz, verbose=False
        )
        dets: list[Detection] = []
        names = getattr(model, "names", {}) or {}
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls.item())
                cls_name = str(names.get(cls_id, cls_id)).lower()
                # If the model already has an "ad" class, keep only ads.
                # Otherwise keep configured ad-like classes (picture/figure).
                if self.ad_names and cls_name not in self.ad_names:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                dets.append(
                    Detection(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        score=round(float(b.conf.item()), 3),
                        source=self.name, label=cls_name or "advertisement",
                    )
                )
        return nms(dets, self.nms_iou)

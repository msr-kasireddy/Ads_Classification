"""Ensemble detector: union of the heuristic + a model backend, then NMS.

Useful as a baseline before you have a fine-tuned model: the model catches
graphic-heavy ads, the heuristic catches cleanly-bordered ads, and NMS removes
duplicates.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..geometry import Detection, nms
from .heuristic import HeuristicAdDetector


class EnsembleAdDetector:
    name = "ensemble"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nms_iou = float(cfg.get("detection", "nms_iou", default=0.40))
        self.heuristic = HeuristicAdDetector(cfg)
        self.model = None
        # the model half is optional; degrade to heuristic-only if unavailable
        kind = cfg.get("detection", "model_for_ensemble", default="doclayout")
        try:
            from .model_backends import ModelAdDetector

            self.model = ModelAdDetector(cfg, kind=kind)
        except Exception:
            self.model = None

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        dets: list[Detection] = list(self.heuristic.detect(image_bgr))
        if self.model is not None:
            try:
                dets += self.model.detect(image_bgr)
            except Exception:
                pass  # keep heuristic results even if the model fails to load
        return nms(dets, self.nms_iou)

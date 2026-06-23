"""Pluggable advertisement detectors."""
from __future__ import annotations

from ..config import Config
from .heuristic import HeuristicAdDetector


def build_detector(cfg: Config):
    """Return a detector instance for the configured backend.

    Model backends are imported lazily so the heuristic backend keeps working
    even when ultralytics / torch are not installed.
    """
    backend = (cfg.get("detection", "backend", default="heuristic") or "heuristic").lower()

    if backend == "heuristic":
        return HeuristicAdDetector(cfg)

    if backend in {"doclayout", "yolov8"}:
        from .model_backends import ModelAdDetector

        return ModelAdDetector(cfg, kind=backend)

    if backend == "ensemble":
        from .ensemble import EnsembleAdDetector

        return EnsembleAdDetector(cfg)

    raise ValueError(f"Unknown detection.backend: {backend!r}")

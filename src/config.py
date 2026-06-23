"""Configuration loading and lightweight access helpers."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    # --- nested getters -------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def page_size_cm(self, publication: str | None) -> tuple[float, float]:
        """Return (width_cm, height_cm) for a publication name, falling back
        to 'default'. Matching is case-insensitive and substring-based so a
        filename like 'sakshi_hyd_2024.jpg' resolves to the sakshi entry."""
        pubs = self.get("publications", default={}) or {}
        name = (publication or "").lower()
        for key, val in pubs.items():
            if key != "default" and key.lower() in name:
                return float(val["page_width_cm"]), float(val["page_height_cm"])
        d = pubs.get("default", {"page_width_cm": 33.0, "page_height_cm": 52.0})
        return float(d["page_width_cm"]), float(d["page_height_cm"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config(raw=_DEFAULTS())
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    merged = _DEFAULTS()
    _deep_update(merged, data)
    return Config(raw=merged)


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _DEFAULTS() -> dict:
    """Built-in defaults so the code runs even without a config.yaml."""
    return copy.deepcopy(
        {
            "publications": {
                "sakshi": {"page_width_cm": 33.0, "page_height_cm": 52.0},
                "eenadu": {"page_width_cm": 33.0, "page_height_cm": 52.0},
                "default": {"page_width_cm": 33.0, "page_height_cm": 52.0},
            },
            "calibration": {
                "mode": "auto_content",
                "dpi": 150,
                "ink_threshold": 200,
                "min_ink_fraction": 0.01,
            },
            "detection": {
                "backend": "heuristic",
                "min_ad_area_cm2": 15.0,
                "max_ad_area_frac": 0.95,
                "nms_iou": 0.40,
                "model_conf": 0.25,
                "yolov8_weights": "models/ads_yolov8.pt",
                "ad_class_names": ["advertisement", "ad", "picture", "figure", "image"],
            },
            "heuristic": {
                "line_len_frac": 0.020,
                "close_frac": 0.012,
                "min_box_aspect": 0.10,
                "max_box_aspect": 10.0,
                "min_fill": 0.04,
            },
            "output": {
                "dir": "data/output",
                "box_color": [0, 0, 255],
                "box_thickness": 4,
                "font_scale": 0.9,
                "draw_area_label": True,
            },
        }
    )

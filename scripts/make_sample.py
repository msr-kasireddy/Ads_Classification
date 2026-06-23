#!/usr/bin/env python3
"""Generate a synthetic broadsheet page (columns of body text + a few boxed
ads) so the pipeline/dashboard can be demoed without real data.

    python scripts/make_sample.py            # -> data/images/sample_sakshi.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_page(w=1650, h=2600) -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 255, np.uint8)
    margin = 60
    cols, gut = 6, 18
    col_w = (w - 2 * margin - (cols - 1) * gut) // cols

    # masthead
    cv2.putText(img, "SAKSHI", (margin, 50), cv2.FONT_HERSHEY_DUPLEX, 2.2,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.line(img, (margin, 70), (w - margin, 70), (0, 0, 0), 2)

    # body text as rows of short "word" dashes with gaps (realistic: body text
    # does NOT form long solid rules, unlike ad borders)
    for c in range(cols):
        x0 = margin + c * (col_w + gut)
        y = 100
        while y < h - margin:
            x = x0
            while x < x0 + col_w - 6:
                word = int(rng.integers(10, 34))
                word = min(word, x0 + col_w - x)
                cv2.line(img, (x, y), (x + word, y), (70, 70, 70), 2)
                x += word + int(rng.integers(5, 11))  # inter-word gap
            y += 13

    # paint a few boxed display ads over the text (these are the targets)
    ads = [
        (margin, h - 760, margin + 3 * col_w + 2 * gut, h - 120),   # big block
        (margin + 3 * (col_w + gut), 700, w - margin, 1150),        # banner
        (margin + 4 * (col_w + gut), 1300, w - margin, 1700),       # square
    ]
    for (x1, y1, x2, y2) in ads:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy - 40), 60, (0, 0, 0), 6)
        cv2.putText(img, "ADVERTISEMENT", (x1 + 20, cy + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "data" / "images"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sample_sakshi.png"
    cv2.imwrite(str(path), make_page())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Route B: small-object recovery for distant litter.

Two zero-training techniques:
1. **Zoom-at-candidate**: when something happens near the ground (a new blob, a
   throw gesture's predicted landing point, a marginal detection), crop that
   region from the ORIGINAL-resolution frame and re-run the litter detectors on
   the crop. A 10x17 px object becomes 100x170 px in the crop — the detector's
   practical threshold.
2. **Tiled inference (SAHI-style)**: slice the frame into overlapping tiles and
   detect per tile, then merge. 2x2 tiles with 20% overlap ~ 2x effective
   magnification, model-agnostic (Akyon et al., ICIP 2022).

Detections are returned in ORIGINAL-frame coordinates so they can be merged with
the main tracker's coordinates.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .channels import _parse_ultralytics


class ZoomDetector:
    """Re-detect litter around candidate points using the native-resolution frame."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        m = cfg["models"]
        self.device = "cpu"
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        self.enabled = bool(m.get("zoom_enable", True))
        self.models: List[Tuple[str, object]] = []
        for name, path in (("litter_a", m.get("litter_a")), ("litter_b", m.get("litter_b"))):
            if path and os.path.exists(path):
                try:
                    mdl = YOLO(path)
                    mdl.to(self.device)
                    self.models.append((name, mdl))
                except Exception as e:
                    print(f"[zoom] model {name} unavailable: {e}")
        b = cfg.get("small_litter", {}) or {}
        self.crop_px = int(b.get("crop_px", 320))        # crop size in ORIGINAL px
        self.upscale = float(b.get("upscale", 2.0))      # crop -> detector input scale
        self.min_conf = float(b.get("min_conf", 0.10))
        self.tiles = int(b.get("tiles", 0))              # 0 = off, 2 = 2x2 slicing
        self.tile_overlap = float(b.get("tile_overlap", 0.2))

    # ------------------------------------------------------------------ zoom
    def zoom_at(self, frame_hi: np.ndarray, point: Tuple[float, float]
                ) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        """Detect around `point` (original-frame coords). -> [(box, conf, source)]"""
        if not self.enabled or frame_hi is None or not self.models:
            return []
        h, w = frame_hi.shape[:2]
        cx, cy = point
        half = self.crop_px / 2.0
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        crop = frame_hi[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        scale = self.upscale
        target = (int(cw * scale), int(ch * scale))
        if target[0] < 32 or target[1] < 32:
            return []
        crop_big = cv2.resize(crop, target, interpolation=cv2.INTER_CUBIC)
        out: List[Tuple[Tuple[float, float, float, float], float, str]] = []
        for name, mdl in self.models:
            try:
                res = mdl.predict(crop_big, verbose=False, device=self.device,
                                   conf=self.min_conf)[0]
            except Exception:
                continue
            sx = (x2 - x1) / target[0]
            sy = (y2 - y1) / target[1]
            for box, conf, _cls, _tid in _parse_ultralytics(res):
                bx1 = x1 + box[0] * sx
                by1 = y1 + box[1] * sy
                bx2 = x1 + box[2] * sx
                by2 = y1 + box[3] * sy
                out.append(((bx1, by1, bx2, by2), conf, name))
        return out

    # ----------------------------------------------------------------- tiles
    def tiled(self, frame_hi: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        """SAHI-style sliced inference over the whole frame (original coords)."""
        if not self.enabled or self.tiles < 2 or frame_hi is None or not self.models:
            return []
        h, w = frame_hi.shape[:2]
        n = self.tiles
        th, tw = h // n, w // n
        ox, oy = int(tw * self.tile_overlap), int(th * self.tile_overlap)
        out: List[Tuple[Tuple[float, float, float, float], float, str]] = []
        seen: List[Tuple[float, float, float, float]] = []
        for iy in range(n):
            for ix in range(n):
                y1 = max(0, iy * th - (oy if iy else 0))
                x1 = max(0, ix * tw - (ox if ix else 0))
                y2 = min(h, (iy + 1) * th + (oy if iy < n - 1 else 0))
                x2 = min(w, (ix + 1) * tw + (ox if ix < n - 1 else 0))
                tile = frame_hi[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                for name, mdl in self.models:
                    try:
                        res = mdl.predict(tile, verbose=False, device=self.device,
                                           conf=self.min_conf)[0]
                    except Exception:
                        continue
                    for box, conf, _c, _t in _parse_ultralytics(res):
                        b = (x1 + box[0], y1 + box[1], x1 + box[2], y1 + box[3])
                        if any(_iou(b, s) > 0.5 for s in seen):
                            continue
                        seen.append(b)
                        out.append((b, conf, name))
        return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

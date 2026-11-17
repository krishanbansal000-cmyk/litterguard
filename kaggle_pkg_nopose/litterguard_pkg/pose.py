"""Pose channel (Route A): release-gesture detection from body keypoints.

The idea (published precedent: illegal-bowling "release frame" detection, and
ETRI 2019 for actual dumping actions): a thrown object leaves the hand when the
arm goes UP past the shoulder and then DOWN again. That motion is measurable at
distances where the object itself is far below any detector's resolution.

This module only measures the GESTURE. The engine turns it into a corroborating
vote when a NEW litter detection appears at the predicted landing point — so a
gesture alone can never create an alert.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# COCO-17 keypoint indices
K_NOSE, K_LSHO, K_RSHO, K_LEL, K_REL, K_LWR, K_RWR = 0, 5, 6, 7, 8, 9, 10


class PoseChannel:
    """Runs a pose model and returns per-person keypoints."""

    def __init__(self, weights: str, cfg: dict):
        from ultralytics import YOLO
        self.device = "cpu"
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def update(self, frame: np.ndarray) -> List[dict]:
        """-> [{'box': (x1,y1,x2,y2), 'kps': {idx: (x, y, conf)}}]"""
        res = self.model.predict(frame, verbose=False, device=self.device,
                                 conf=0.2, classes=[0])[0]
        out: List[dict] = []
        kps = getattr(res, "keypoints", None)
        boxes = getattr(res, "boxes", None)
        if kps is None or boxes is None or len(boxes) == 0:
            return out
        data = kps.data.cpu().numpy() if hasattr(kps, "data") else np.array([])
        for i, box in enumerate(boxes.xyxy.tolist()):
            if i >= len(data):
                break
            pts = data[i]
            kd = {}
            for j, p in enumerate(pts):
                if j >= len(p):
                    break
                x, y, c = float(p[0]), float(p[1]), float(p[2])
                if c > 0.25:
                    kd[j] = (x, y, c)
            out.append({"box": tuple(box), "kps": kd})
        return out


class GestureTracker:
    """Per-person gesture state machine: ARMED (object in hand) -> RAISE -> DROP."""

    def __init__(self, cfg: dict):
        g = cfg.get("pose_gesture", {}) or {}
        self.g = g
        self.enable = bool(g.get("enable", True))
        self.arm_radius_frac = float(g.get("arm_radius_frac", 0.30))
        self.raise_thresh = float(g.get("raise_thresh", 0.10))
        self.drop_thresh = float(g.get("drop_thresh", -0.15))
        self.release_window_s = float(g.get("release_window_s", 1.5))
        self.ground_offset_frac = float(g.get("ground_offset_frac", 0.35))
        self.state: Dict[int, dict] = {}
        self.throws: Deque[dict] = deque(maxlen=32)

    def _key(self, track_id: int) -> dict:
        st = self.state.get(track_id)
        if st is None:
            st = {"armed": False, "arm_t": 0.0, "raised": False, "raise_t": 0.0,
                  "h": 0.0, "wrist": None}
            self.state[track_id] = st
        return st

    def update(self, t: float, track_id: int, pose: Optional[dict],
               litter_dets: List[dict]) -> None:
        """Feed one tracked person: their keypoints + current litter detections."""
        if not self.enable or pose is None:
            return
        kps = pose.get("kps") or {}
        box = pose.get("box")
        if not kps or box is None:
            return
        shoulders = [kps[j] for j in (K_LSHO, K_RSHO) if j in kps]
        wrists = [kps[j] for j in (K_LWR, K_RWR) if j in kps]
        if not shoulders or not wrists:
            return
        # pick the arm with the more confident wrist
        wrist = max(wrists, key=lambda p: p[2])
        shoulder_y = min(p[1] for p in shoulders)
        height = max(1.0, float(box[3] - box[1]))
        h = (shoulder_y - wrist[1]) / height      # >0 arm raised, <0 arm down

        st = self._key(track_id)
        st["h"] = h
        st["wrist"] = (wrist[0], wrist[1])

        # ARMED: a litter detection near the hand
        if not st["armed"]:
            r = self.arm_radius_frac * height
            for d in litter_dets:
                cx = (d["box"][0] + d["box"][2]) / 2.0
                cy = (d["box"][1] + d["box"][3]) / 2.0
                if abs(cx - wrist[0]) <= r and abs(cy - wrist[1]) <= r:
                    st["armed"] = True
                    st["arm_t"] = t
                    break

        # RAISE then DROP => release
        if st["armed"] and not st["raised"] and h > self.raise_thresh:
            st["raised"] = True
            st["raise_t"] = t
        if st["raised"] and h < self.drop_thresh and (t - st["raise_t"]) <= self.release_window_s:
            ground = (wrist[0], wrist[1] + self.ground_offset_frac * height)
            self.throws.append({"t": t, "point": ground, "track_id": track_id,
                                "height": height})
            st["armed"] = False
            st["raised"] = False

    def gesture_for(self, t: float, point: Tuple[float, float],
                    px_per_m: float) -> bool:
        """Is a NEW detection at `point` explained by a recent throw gesture?"""
        radius_m = float(self.g.get("ground_radius_m", 1.5))
        window = float(self.g.get("confirm_window_s", 2.5))
        r_px = radius_m * max(px_per_m, 1e-6)
        for th in list(self.throws):
            if 0 <= t - th["t"] <= window:
                if abs(th["point"][0] - point[0]) <= r_px and \
                   abs(th["point"][1] - point[1]) <= r_px:
                    return True
        return False

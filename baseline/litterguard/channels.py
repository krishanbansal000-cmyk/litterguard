"""Detection channels: persons/vehicles, two litter detectors, background
static-foreground, and the slow GroundingDINO anchor thread.

Every channel is a pre-trained, publicly downloadable model or a classical CV
method — nothing here is trained by us. Tracking uses Ultralytics' built-in
ByteTrack (persist mode).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

COCO_VEHICLES = [1, 2, 3, 5, 7]  # bicycle car motorcycle bus truck


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def xyxy_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class Det:
    """Backend-agnostic detection."""
    box: Tuple[float, float, float, float]      # xyxy pixels
    score: float
    label: str
    source: str                                  # "person" | "litter_a" | "litter_b" | "gd"
    track_id: Optional[int] = None


def _parse_ultralytics(result) -> List[Tuple[Tuple[float, ...], float, int, Optional[int]]]:
    """-> [(box, score, class_id, track_id_or_None)]"""
    out = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
    for xyxy, conf, cls, tid in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                    boxes.cls.tolist(), ids):
        out.append((tuple(xyxy), float(conf), int(cls),
                    int(tid) if tid is not None else None))
    return out


def resolve_device(cfg: dict) -> str:
    """runtime.device: auto (cuda if available) | cpu | cuda"""
    import torch
    d = (cfg.get("runtime") or {}).get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


# --------------------------------------------------------------------------
# Channel A + vehicles: YOLO11n (COCO) + Ultralytics ByteTrack (persist)
# --------------------------------------------------------------------------
class PersonVehicleChannel:
    """Persons and vehicles from a stock COCO model, tracked by ByteTrack."""

    def __init__(self, weights: str, cfg: dict):
        from ultralytics import YOLO
        self.device = resolve_device(cfg)
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.tracker_yaml = cfg["models"].get("person_tracker_yaml", "models/bytetrack_lg.yaml")
        self.person_conf = float(cfg["models"].get("person_conf", 0.25))
        ve = cfg.get("vehicle_eject", {})
        self.vehicle_classes = [int(c) for c in ve.get("vehicle_classes", COCO_VEHICLES)] \
            if ve.get("enable", True) else []
        self.classes = [0] + self.vehicle_classes
        self.person_heights: Deque[float] = deque(maxlen=120)  # px/m calibration

    def update(self, frame: np.ndarray, t: float) -> Tuple[List[Det], List[Det]]:
        res = self.model.track(frame, persist=True, tracker=self.tracker_yaml,
                               classes=self.classes, conf=self.person_conf,
                               verbose=False, device=self.device)[0]
        persons: List[Det] = []
        vehicles: List[Det] = []
        for box, score, cls, tid in _parse_ultralytics(res):
            d = Det(box, score, "person" if cls == 0 else "vehicle",
                    "person", tid)
            if cls == 0:
                h = box[3] - box[1]
                if h > 20:
                    self.person_heights.append(float(h))
                persons.append(d)
            else:
                vehicles.append(d)
        return persons, vehicles

    @property
    def px_per_m(self) -> float:
        if not self.person_heights:
            return 40.0  # sane default for 960px scenes
        return float(np.median(self.person_heights)) / 1.7


# --------------------------------------------------------------------------
# Channels B + C: litter detectors (community YOLOv8 + zero-shot YOLO-World)
# --------------------------------------------------------------------------
class LitterChannel:
    """Two independent pre-trained litter detectors, source-tagged detections."""

    def __init__(self, weights_a: str, weights_b: str, cfg: dict):
        from ultralytics import YOLO
        m = cfg["models"]
        self.device = resolve_device(cfg)
        self.model_a = YOLO(weights_a)
        self.model_a.to(self.device)
        self.vocab: List[str] = list(m["world_vocab"])
        self.model_b = YOLO(weights_b)
        self.model_b.to(self.device)
        self.conf_a = float(m.get("litter_a_conf", 0.30))
        self.conf_b = float(m.get("world_conf", 0.24))
        # extra community litter detectors = additional INDEPENDENT votes
        self.extra: List[Tuple[str, object]] = []
        for i, path in enumerate(m.get("extra_detectors", []) or []):
            if os.path.exists(path):
                try:
                    mdl = YOLO(path)
                    mdl.to(self.device)
                    self.extra.append((f"litter_{chr(ord('c') + i)}", mdl))
                    print(f"[litter] extra detector {chr(ord('c') + i)}: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[litter] extra detector failed {path}: {e}")
        self._b_ready = False
        try:
            self.model_b.set_classes(self.vocab)  # zero-shot vocab at runtime
            self._b_ready = True
        except Exception as e:
            print(f"[litter_b] YOLO-World vocab failed ({e}); COCO fallback")

    def update(self, frame: np.ndarray) -> List[Det]:
        out: List[Det] = []
        for box, score, _cls, _tid in _parse_ultralytics(
                self.model_a.predict(frame, verbose=False, device=self.device,
                                     conf=self.conf_a)[0]):
            out.append(Det(box, score, "litter", "litter_a"))
        rb = self.model_b.predict(frame, verbose=False, device=self.device,
                                  conf=self.conf_b)[0]
        if self._b_ready:
            for box, score, _cls, _tid in _parse_ultralytics(rb):
                out.append(Det(box, score, "litter", "litter_b"))
        else:  # COCO fallback: litter-ish classes only
            # COCO fallback classes (bottle 39, cup 41, wine glass 40, backpack 24,
            # handbag 26, suitcase 28)
            for box, score, cls, _tid in _parse_ultralytics(rb):
                if cls in (24, 26, 28, 39, 40, 41):
                    out.append(Det(box, score, "litter", "litter_b"))
        for src, mdl in self.extra:
            try:
                re_ = mdl.predict(frame, verbose=False, device=self.device, conf=0.30)[0]
                for box, score, _cls, _tid in _parse_ultralytics(re_):
                    out.append(Det(box, score, "litter", src))
            except Exception:
                pass
        return out


# --------------------------------------------------------------------------
# Channel D: MOG2 static-foreground blobs (classical, training-free)
# --------------------------------------------------------------------------
class BackgroundChannel:
    """Physical-object proof: a pixel region that is foreground and stays put."""

    def __init__(self, cfg: dict, history: int = 300, var_threshold: int = 24):
        self.mog = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                      detectShadows=True)
        self.static_mask = np.zeros((10, 10), np.float32)
        self.frame_count = 0
        self.min_frames = int((cfg.get("logic") or {}).get("bg_min_frames", 25))

    def reset(self):
        self.static_mask = np.zeros_like(self.static_mask, np.float32)
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> np.ndarray:
        fg = self.mog.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        if self.static_mask.shape != fg.shape:
            self.static_mask = np.zeros_like(fg, np.float32)
        self.static_mask = self.static_mask * 0.92 + (fg > 200).astype(np.float32) * 0.08
        self.frame_count += 1
        return fg

    def blob_at(self, box: Tuple[float, float, float, float], ratio: float = 0.25) -> bool:
        """True when >=ratio of the box area is a mature static-foreground blob."""
        if self.frame_count < self.min_frames:
            return False
        h, w = self.static_mask.shape[:2]
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                          int(min(w, box[2])), int(min(h, box[3])))
        if x2 <= x1 or y2 <= y1:
            return False
        region = self.static_mask[y1:y2, x1:x2]
        return float((region > 0.55).mean()) >= ratio


# --------------------------------------------------------------------------
# Channel E: GroundingDINO slow anchor (own thread, latest-frame handoff)
# --------------------------------------------------------------------------
@dataclass
class GDVote:
    box: Tuple[float, float, float, float]
    score: float
    label: str
    t: float


class GroundingDINOChannel:
    """Zero-shot text-prompted detector as the slow accuracy anchor."""

    def __init__(self, model_id: str, cfg: dict):
        self.cfg = cfg["models"]
        self.device = resolve_device(cfg)
        self.votes: Deque[GDVote] = deque(maxlen=200)
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ok = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        def worker():
            try:
                from transformers import AutoProcessor, GroundingDinoForObjectDetection
                from PIL import Image
                import torch
                proc = AutoProcessor.from_pretrained(self.cfg["grounding_dino"])
                model = GroundingDinoForObjectDetection.from_pretrained(self.cfg["grounding_dino"])
                model.eval().to(self.device)
                self._ok = True
                print("[GD] anchor ready")
                interval = float(self.cfg.get("gd_interval_s", 4.0))
                while not self._stop.is_set():
                    snap = None
                    with self._lock:
                        if self._latest is not None:
                            snap = self._latest
                            self._latest = None
                    if snap is None:
                        time.sleep(0.3)
                        continue
                    frame, t = snap
                    with torch.no_grad():
                        inputs = proc(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                      text=self.cfg["gd_prompt"], return_tensors="pt")
                        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                                  for k, v in inputs.items()}
                        out = model(**inputs)
                        # transformers 5.x renamed box_threshold -> threshold
                        try:
                            res = proc.post_process_grounded_object_detection(
                                out, input_ids,
                                threshold=float(self.cfg.get("gd_box_threshold", 0.30)),
                                text_threshold=float(self.cfg.get("gd_text_threshold", 0.25)),
                                target_sizes=[frame.shape[:2]])[0]
                        except TypeError:
                            res = proc.post_process_grounded_object_detection(
                                out, input_ids,
                                box_threshold=float(self.cfg.get("gd_box_threshold", 0.30)),
                                text_threshold=float(self.cfg.get("gd_text_threshold", 0.25)),
                                target_sizes=[frame.shape[:2]])[0]
                    labels = res["text_labels"] if "text_labels" in res else res["labels"]
                    for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(),
                                               labels):
                        with self._lock:
                            self.votes.append(GDVote(tuple(box), float(score), str(lab), t))
                    time.sleep(max(0.2, interval))
            except Exception as e:  # anchor is optional
                print(f"[GD] disabled ({e.__class__.__name__}: {e})")

        self._thread = threading.Thread(target=worker, daemon=True, name="gd-anchor")
        self._thread.start()

    def submit(self, frame: np.ndarray, t: float):
        with self._lock:
            self._latest = (frame.copy(), t)

    def stop(self):
        self._stop.set()

    @property
    def available(self) -> bool:
        return self._ok

    def vote_near(self, box: Tuple[float, float, float, float], t: float,
                  ttl: float, iou_min: float = 0.25) -> Optional[GDVote]:
        best = None
        with self._lock:
            for v in self.votes:
                if t - v.t > ttl or t < v.t - 2.0:
                    continue
                if xyxy_iou(box, v.box) >= iou_min and (best is None or v.score > best.score):
                    best = v
        return best

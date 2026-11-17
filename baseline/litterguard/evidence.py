"""Evidence packs: pre/post event clips from a ring buffer, culprit & litter
crops, annotated keyframe, and a JSONL event log per run."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .engine import Event


class EvidenceWriter:
    def __init__(self, out_root: str, source_name: str, cfg: dict, fps: float, size: Tuple[int, int]):
        self.cfg = cfg["evidence"]
        self.fps = max(fps, 1.0)
        self.size = size
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(out_root, f"{stamp}_{os.path.splitext(source_name)[0]}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl = os.path.join(self.run_dir, "events.jsonl")
        self._pre = int(self.cfg["pre_event_s"] * fps)
        self.buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self._pre + 10)
        self._open: List[Tuple[Event, str, cv2.VideoWriter, float]] = []
        self._n = 0

    # called every processed frame
    def push(self, t: float, frame: np.ndarray, annotated: np.ndarray):
        self.buffer.append((t, frame))
        # feed open recordings
        still: List = []
        for ev, path, vw, end_t in self._open:
            vw.write(annotated if annotated is not None else frame)
            if t < end_t:
                still.append((ev, path, vw, end_t))
            else:
                vw.release()
                print(f"[evidence] clip closed: {path}")
        self._open = still

    # called by pipeline when engine fires
    def begin(self, ev: Event, snapshot_frame: Optional[np.ndarray], culprit_box, litter_box):
        self._n += 1
        name = f"ev{self._n:03d}_{ev.type}_{time.strftime('%H%M%S')}"
        base = os.path.join(self.run_dir, name)
        clip = base + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(clip, fourcc, self.fps, (self.size[1], self.size[0]))
        if not vw.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vw = cv2.VideoWriter(clip + ".avi", fourcc, self.fps, (self.size[1], self.size[0]))
            clip = clip + ".avi"
        for (bt, bf), _ in zip(self.buffer, range(len(self.buffer))):
            if bt >= ev.t - float(self.cfg["pre_event_s"]):
                vw.write(bf)
        self._open.append((ev, clip, vw, ev.t + float(self.cfg["post_event_s"])))

        # crops
        if snapshot_frame is not None:
            h, w = snapshot_frame.shape[:2]
            if culprit_box:
                x1, y1, x2, y2 = _clamp(culprit_box, w, h, pad=0.12)
                cv2.imwrite(f"{base}_culprit.jpg", snapshot_frame[y1:y2, x1:x2],
                            [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_quality"])])
            x1, y1, x2, y2 = _clamp(litter_box, w, h, pad=0.35)
            cv2.imwrite(f"{base}_litter.jpg", snapshot_frame[y1:y2, x1:x2],
                        [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_quality"])])

        rec = {"event": name, "type": ev.type, "t": round(ev.t, 2),
               "votes": ev.votes, "litter_box": [round(v, 1) for v in ev.litter_box],
               "culprit": ev.culprit, "vehicle_box": ev.vehicle_box and [round(v, 1) for v in ev.vehicle_box],
               "extras": ev.extras, "clip": os.path.basename(clip)}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[EVENT] {ev.type} @t={ev.t:.1f}s votes={ev.votes} culprit="
              f"{ev.culprit and ev.culprit['track_id']} -> {os.path.basename(clip)}")
        return clip

    def close(self):
        for _, _, vw, _ in self._open:
            vw.release()
        self._open = []


def _clamp(box, w, h, pad: float = 0.0):
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - pw)), int(max(0, y1 - ph)),
            int(min(w, x2 + pw)), int(min(h, y2 + ph)))

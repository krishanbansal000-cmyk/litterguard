"""Pipeline orchestrator: source -> channels -> engine -> evidence."""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from .channels import (BackgroundChannel, GroundingDINOChannel, LitterChannel,
                       PersonVehicleChannel, center)
from .engine import LitterEngine
from .evidence import EvidenceWriter

COLORS = {"person": (60, 200, 60), "litter": (60, 120, 255),
          "event": (0, 0, 255), "vehicle": (200, 160, 40)}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_source(src: str):
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def resize_keep(frame: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = max_edge / float(max(h, w))
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return frame


class Pipeline:
    def __init__(self, cfg: dict, source_override: Optional[str] = None,
                 show: bool = False, record: bool = False):
        self.cfg = cfg
        src = source_override or cfg["source"]["input"]
        self.src = src
        self.cap = open_source(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {src}")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("source produced no first frame")
        frame = resize_keep(frame, int(cfg["source"]["resize_max"]))
        self.frame_shape = frame.shape
        self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.proc_interval = 1.0 / float(cfg["source"]["proc_fps"])

        import torch
        torch.set_num_threads(int(cfg["runtime"].get("torch_threads", 8)))
        self.device = "cuda" if (cfg["runtime"].get("device", "auto") == "auto" and torch.cuda.is_available()) else cfg["runtime"].get("device", "auto")

        print(f"[init] loading models (device={self.device}) ...")
        self.persons = PersonVehicleChannel(cfg["models"]["person"], cfg)
        print("[init] person/vehicle channel ready")
        self.litter = LitterChannel(cfg["models"]["litter_a"], cfg["models"]["litter_b"], cfg)
        print(f"[init] litter channels ready (world vocab ok={self.litter._b_ready})")
        self.bg = BackgroundChannel(cfg)
        self.gd = GroundingDINOChannel(cfg["models"]["grounding_dino"], cfg)
        self.gd.start()
        print("[init] GroundingDINO anchor thread started (loads in background)")

        self.engine = LitterEngine(cfg, on_event=None)
        self.engine.frame_wh = (self.frame_shape[1], self.frame_shape[0])
        name = os.path.basename(str(src)) or "webcam"
        self.evidence = EvidenceWriter(cfg["evidence"]["out_dir"], name, cfg,
                                       float(cfg["source"]["proc_fps"]), self.frame_shape)
        self.show = show or bool(cfg["runtime"]["show_window"])
        self.record = record
        self._rec_writer = None
        self._last_frame: Optional[np.ndarray] = None
        self._litter_dets = []
        self._last_litter_t = -1e9
        self._last_extra_t = -1e9
        self._last_pose_t = -1e9
        self.pose = None
        self.gesture = None
        if (cfg.get("pose_gesture") or {}).get("enable", False):
            from .pose import GestureTracker, PoseChannel
            self.gesture = GestureTracker(cfg)
            try:
                self.pose = PoseChannel(cfg["pose_gesture"]["model"], cfg)
                print(f"[init] pose gesture channel ready ({cfg['pose_gesture']['model']})")
            except Exception as e:
                print(f"[init] pose channel unavailable: {e}")
                self.pose = None
        self._stats = {"frames": 0, "proc": 0, "t0": None, "events": 0}

    # ------------------------------------------------------------------ restart
    def restart(self, source: str):
        """Re-target an already-loaded pipeline at a new video: models stay in
        memory; per-video state (engine, background, evidence, GD votes) resets.
        ~3x faster than re-constructing Pipeline for every short clip."""
        import time as _t
        self.cap.release()
        self.src = source
        self.cap = open_source(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {source}")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("source produced no first frame")
        frame = resize_keep(frame, int(self.cfg["source"]["resize_max"]))
        self.frame_shape = frame.shape
        self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        from .engine import LitterEngine
        self.engine = LitterEngine(self.cfg, on_event=None)
        self.engine.frame_wh = (self.frame_shape[1], self.frame_shape[0])
        name = os.path.basename(str(source)) or "webcam"
        self.evidence = EvidenceWriter(self.cfg["evidence"]["out_dir"], name, self.cfg,
                                       float(self.cfg["source"]["proc_fps"]), self.frame_shape)
        self.bg.reset()
        self.persons.reset()
        if self.gesture is not None:
            self.gesture.state.clear()
            self.gesture.throws.clear()
        self._last_pose_t = -1e9
        with self.gd._lock:
            self.gd.votes.clear()
        self._rec_writer = None
        self._stats = {"frames": 0, "proc": 0, "t0": None, "events": 0}
        self._last_frame = None
        self._litter_dets = []
        self._last_litter_t = -1e9
        self._last_extra_t = -1e9

    # ------------------------------------------------------------------ run
    def run(self):
        cfg = self.cfg
        interval = self.proc_interval
        last_proc_t = -1e9
        clock = 0.0                 # engine clock: video time for files, wall for live
        use_wall = str(self.src).isdigit() or str(self.src).lower().startswith("rtsp")
        frame = None
        print(f"[run] source={self.src} proc_fps={cfg['source']['proc_fps']} "
              f"shape={self.frame_shape[1]}x{self.frame_shape[0]}")
        while True:
            ok, f = self.cap.read()
            if not ok:
                break
            pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if use_wall or pos <= 0:
                clock = time.time()
            else:
                clock = pos
            f = resize_keep(f, int(cfg["source"]["resize_max"]))
            frame = f
            if self._stats["t0"] is None:
                self._stats["t0"] = time.time()
            if clock - last_proc_t >= interval:
                last_proc_t = clock
                self._process(frame, clock)
                self._stats["proc"] += 1
            if self.show:
                cv2.imshow("LitterGuard", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        self.finish()

    # -------------------------------------------------------------- one step
    def _process(self, frame: np.ndarray, t: float):
        self._last_frame = frame
        persons, vehicles = self.persons.update(frame, t)
        # not need full rate — reuse the last detections in between. The extra
        # corroborating detectors run even slower (they only add votes).
        litter_fps = float(self.cfg["source"].get("litter_fps", 0))
        extra_fps = float(self.cfg["source"].get("extra_fps", 1.0))
        run_extras = extra_fps <= 0 or (t - self._last_extra_t) >= (1.0 / extra_fps)
        if litter_fps <= 0 or (t - self._last_litter_t) >= (1.0 / litter_fps):
            self._litter_dets = self.litter.update(frame, run_extras=run_extras)
            self._last_litter_t = t
            if run_extras:
                self._last_extra_t = t
        litter_dets = self._litter_dets
        fg = self.bg.update(frame)
        self.gd.submit(frame, t)

        # --- Route A: pose gesture channel (release detection) ----------------
        if self.pose is not None:
            pose_fps = float(self.cfg["pose_gesture"].get("pose_fps", 2.5))
            if pose_fps <= 0 or (t - self._last_pose_t) >= (1.0 / pose_fps):
                self._last_pose_t = t
                poses = self.pose.update(frame)
                dets_as_dict = [{"box": d.box} for d in litter_dets]
                for p in poses:
                    pc = ((p["box"][0] + p["box"][2]) / 2.0,
                          (p["box"][1] + p["box"][3]) / 2.0)
                    best = None
                    for pd in persons:
                        if pd.track_id is None:
                            continue
                        dc = ((pd.box[0] + pd.box[2]) / 2.0,
                              (pd.box[1] + pd.box[3]) / 2.0)
                        d = float(np.hypot(pc[0] - dc[0], pc[1] - dc[1]))
                        if best is None or d < best[0]:
                            best = (d, pd.track_id)
                    if best and best[0] < 90:
                        self.gesture.update(t, best[1], p, dets_as_dict)

        fired = self.engine.update(t, persons, vehicles, litter_dets,
                                   self.bg, self.gd, self.persons.px_per_m,
                                   gesture=self.gesture)

        annotated = frame.copy()
        _draw(annotated, persons, vehicles, litter_dets, fired)

        for item in fired:
            ev = item["event"]
            culprit_box = ev.culprit and tuple(ev.culprit["box"])
            self.evidence.begin(ev, frame, culprit_box, ev.litter_box)
            self._stats["events"] += 1

        self.evidence.push(t, frame, annotated)
        if self.record:
            if self._rec_writer is None:
                path = os.path.join(self.evidence.run_dir, "annotated_full.mp4")
                self._rec_writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.cfg["source"]["proc_fps"]),
                    (frame.shape[1], frame.shape[0]))
            self._rec_writer.write(annotated)

    # ---------------------------------------------------------------- finish
    def finish(self):
        # end-of-stream: final relaxed evaluation of pending litter tracks
        try:
            last_t = max((tr.last_t for tr in self.engine.litter.values()), default=0.0)
            fired = self.engine.flush(last_t, bg=self.bg, gd=self.gd)
            for item in fired:
                ev = item["event"]
                culprit_box = ev.culprit and tuple(ev.culprit["box"])
                self.evidence.begin(ev, self._last_frame, culprit_box, ev.litter_box)
                self._stats["events"] += 1
        except Exception as e:
            print(f"[flush] skipped ({e.__class__.__name__}: {e})")
        self.gd.stop()
        self.evidence.close()
        if self._rec_writer is not None:
            self._rec_writer.release()
        if self.show:
            cv2.destroyAllWindows()
        s = self._stats
        dt = (s["t0"] and time.time() - s["t0"]) or 1.0
        print(f"\n[summary] processed={s['proc']} frames wall={dt:.1f}s "
              f"({s['proc']/max(dt,1e-9):.1f} fps) events={s['events']} "
              f"guards_suppressed={len(self.engine.suppressed)} "
              f"-> {self.evidence.run_dir}")
        for g in self.engine.suppressed:
            print(f"  [guard] t={g['t']}s track={g['track']} reason={g['reason']}")


# --------------------------------------------------------------------------
def _draw(img, persons, vehicles, litter_dets, fired):
    for p in persons:
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["person"], 2)
        cv2.putText(img, f"P{p.track_id}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["person"], 1)
    for v in vehicles:
        x1, y1, x2, y2 = map(int, v.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["vehicle"], 2)
        cv2.putText(img, "vehicle", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["vehicle"], 1)
    for d in litter_dets:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["litter"], 1)
        cv2.putText(img, d.source, (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["litter"], 1)
    for item in fired:
        ev = item["event"]
        x1, y1, x2, y2 = map(int, ev.litter_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["event"], 3)
        cv2.putText(img, f"{ev.type.upper()} votes={len(ev.votes)}",
                    (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLORS["event"], 2)
        if ev.culprit:
            cx1, cy1, cx2, cy2 = map(int, ev.culprit["box"])
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), COLORS["event"], 3)

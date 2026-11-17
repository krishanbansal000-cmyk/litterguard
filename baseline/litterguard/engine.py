"""Temporal logic engine — the "brain". No ML training, pure state machines:

  Litter track:  NEW + STATIONARY + multi-source AGREEMENT + physical blob
                 + culprit PROXIMITY + exclusions  =>  littering event
  Chain track:   person HOLDING litter -> RELEASE -> LANDED  => fast high-confidence event
  Vehicle rule:  litter spawns with no person near but a vehicle near => vehicle_eject
  Suppression:   pickup (track dies near person), bin zones, cooldowns, bootstrap.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .channels import COCO_VEHICLES, Det, GDVote, xyxy_iou, center


# --------------------------------------------------------------------------
IOU_STABILITY_MIN = 0.35   # mean consecutive-box IoU that counts as "stationary"


@dataclass
class LitterTrack:
    track_id: int
    first_t: float
    last_t: float
    boxes: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=60))
    centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))
    sources: set = field(default_factory=set)
    pre_existing: bool = False
    fired: bool = False
    picked: bool = False
    chain_armed: bool = False
    chain_source: Optional[str] = None
    median_box: Optional[Tuple[float, float, float, float]] = None

    @property
    def age(self) -> float:
        return self.last_t - self.first_t

    @property
    def spawn_center(self) -> Tuple[float, float]:
        return self.centers[0] if self.centers else (0.0, 0.0)

    @property
    def stationary(self) -> bool:
        """Stationary = box stays in the same place. Scale-free measure:
        mean IoU between consecutive recent boxes (robust to edge flicker and
        to a person shifting weight next to the item), with the smoothed-centroid
        sigma as a fallback."""
        if len(self.centers) < 6:
            return False
        bs = list(self.boxes)[-12:]
        ious = [xyxy_iou(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]
        mean_iou = (sum(ious) / len(ious)) if ious else 0.0
        sigma = _smoothed_sigma(list(self.centers))
        return mean_iou >= IOU_STABILITY_MIN or sigma < 20.0


# --------------------------------------------------------------------------
@dataclass
class Event:
    t: float
    type: str                      # "drop" | "chain_drop" | "vehicle_eject"
    litter_box: Tuple[float, float, float, float]
    votes: List[str]
    culprit: Optional[Dict] = None  # {"track_id", "box", "distance_m", "snapshot_box"}
    vehicle_box: Optional[Tuple[float, float, float, float]] = None
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
class LitterEngine:
    def __init__(self, cfg: dict, on_event: Callable[[Event, np.ndarray], None]):
        self.cfg = cfg
        self.logic = cfg["logic"]
        self.on_event = on_event

        # litter tracker state
        self._next_id = 1
        self.litter: Dict[int, LitterTrack] = {}
        self.person_hist: Dict[int, Deque[Tuple[float, Tuple[float, float], Tuple[float, ...]]]] = {}
        self.armed_persons: Dict[int, Tuple[float, Tuple[float, float, float, float]]] = {}
        self.hold_accum: Dict[int, float] = {}
        self.events: List[Event] = []
        self.suppressed: List[dict] = []   # guard rejections, for transparency
        self.frame_wh: Optional[Tuple[int, int]] = None  # (W, H), set by pipeline
        self.t0: Optional[float] = None
        self.px_per_m: float = 40.0

    # ------------------------------------------------------------------ utils
    def _dist_m(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]) / max(self.px_per_m, 1e-6))

    def _in_bin_zone(self, p: Tuple[float, float]) -> bool:
        zones = self.logic.get("bin_zones") or []
        for poly in zones:
            poly = np.array(poly, np.float32)
            if cv2_point_in_poly(p, poly):
                return True
        return False

    def _near_recent_event(self, p: Tuple[float, float], t: float) -> bool:
        cd_s = float(self.logic["event_cooldown_s"])
        cd_r = float(self.logic["event_cooldown_radius_m"])
        for e in self.events:
            if t - e.t < cd_s and self._dist_m(p, center(e.litter_box)) < cd_r:
                return True
        return False

    # ------------------------------------------------------------ main update
    def update(self, t: float, persons: List[Det], vehicles: List[Det],
               litter_dets: List[Det], bg, gd, px_per_m: float) -> List[dict]:
        if self.t0 is None:
            self.t0 = t
        self.px_per_m = px_per_m
        bootstrap_end = self.t0 + float(self.logic["bootstrap_s"])

        # 1) person history ------------------------------------------------
        for p in persons:
            self.person_hist.setdefault(p.track_id if p.track_id is not None else id(p),
                                        deque(maxlen=200)).append((t, center(p.box), p.box))

        # 2) held->release arming (chain) ----------------------------------
        if self.cfg["held_release"]["enable"]:
            arm_iou = float(self.cfg["held_release"]["arm_iou_min"])
            hold_need = float(self.cfg["held_release"]["hold_arm_s"])
            touching: Dict[int, Det] = {}
            for p in persons:
                for d in litter_dets:
                    if xyxy_iou(p.box, d.box) >= arm_iou or _point_in_box(center(d.box), p.box):
                        touching[p.track_id] = d
                        break
            for pid, d in touching.items():
                prev = self.hold_accum.get(pid, 0.0)
                self.hold_accum[pid] = prev + _dt_estimate(self.person_hist.get(pid))
                if self.hold_accum[pid] >= hold_need:
                    self.armed_persons[pid] = (t, d.box, d.source)
            for pid in list(self.hold_accum):
                if pid not in touching:
                    self.hold_accum.pop(pid, None)

        # 3) litter tracker update ----------------------------------------
        for d in litter_dets:
            best_id, best_score = None, 0.25
            for tid, tr in self.litter.items():
                if t - tr.last_t > 3.0:
                    continue
                # match against recent boxes (handles detector dropout jitter)
                for prev in list(tr.boxes)[-3:]:
                    iou = xyxy_iou(d.box, prev)
                    if iou > best_score:
                        best_id, best_score = tid, iou
            if best_id is None:  # center fallback for jittery small items
                c = center(d.box)
                for tid, tr in self.litter.items():
                    if t - tr.last_t > 3.0:
                        continue
                    if any(_dist(c, pc) < 60 for pc in list(tr.centers)[-3:]):
                        best_id = tid
                        break
            if best_id is None:
                tr = LitterTrack(self._next_id, t, t)
                self._next_id += 1
                tr.boxes.append(d.box)
                tr.centers.append(center(d.box))
                tr.sources.add(d.source)
                tr.pre_existing = t < bootstrap_end
                self.litter[tr.track_id] = tr
            else:
                tr = self.litter[best_id]
                tr.last_t = t
                tr.boxes.append(d.box)
                tr.centers.append(center(d.box))
                tr.sources.add(d.source)

        # 4) evaluate tracks ------------------------------------------------
        fired_now: List[dict] = []
        for tid, tr in list(self.litter.items()):
            # pickup suppression: died while a person was near
            if t - tr.last_t > float(self.logic["pickup_suppress_s"]) and not tr.fired:
                near = self._nearest_person(tr, t, window=float(self.logic["pickup_suppress_s"]) + 3)
                if near and near[1] < self.logic["proximity_radius_m"]:
                    tr.picked = True
                continue
            if tr.fired or tr.picked or tr.pre_existing or len(tr.boxes) < 8:
                continue

            box = tuple(np.median(np.array(list(tr.boxes)[-15:]), axis=0).tolist())
            tr.median_box = box
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < float(self.logic["min_litter_area_px"]):
                continue

            # Guard 1: an over-sized box is a scene/person blob, not a litter item
            frame_area = float(self.frame_wh[0] * self.frame_wh[1]) if self.frame_wh else 518400.0
            if area > float(self.logic.get("max_litter_area_frac", 0.35)) * frame_area:
                tr.fired = True  # consume silently: known-bad detector mode
                self.suppressed.append({"t": round(t, 1), "track": tid,
                                        "reason": "oversized_box", "area_px": int(area)})
                continue

            # Guard 2: box overlapping a tracked person = held item / the person
            # themselves (self-attribution) — not ground litter
            pmax = float(self.logic.get("person_containment_max_iou", 0.45))
            if any(xyxy_iou(box, b) > pmax for (_pid, _d, b) in self._persons_near(tr, t)):
                tr.fired = True  # consume silently
                self.suppressed.append({"t": round(t, 1), "track": tid,
                                        "reason": "person_overlap"})
                continue

            chain = self._chain_link(tr, t)
            age_req = float(self.logic["fast_chain_age_s"]) if chain else float(self.logic["stationary_min_age_s"])
            if tr.age < age_req or not tr.stationary:
                continue

            spawn = tr.spawn_center
            if self._in_bin_zone(spawn) or self._near_recent_event(spawn, t):
                tr.fired = True   # consume, stay silent
                continue

            # agreement votes
            votes = set(tr.sources)
            gd_v: Optional[GDVote] = None
            if gd is not None:
                gd_v = gd.vote_near(box, t, ttl=float(self.cfg["models"].get("gd_vote_ttl_s", 45)))
                if gd_v is not None:
                    votes.add("grounding_dino")
            # held->release->landed chain is *support* (shortens the wait), NOT an
            # independent vote (it derives from the same detector's detections —
            # counting it would double-count, per external audit)
            if chain and tr.chain_source:
                pass
            # a very large litter detection from the litter-trained detector is
            # unambiguous bulk waste — let it count as its own second vote
            if area >= float(self.logic.get("strong_area_px", 60000)) and "litter_a" in votes:
                votes.add("large_object")

            # physical proof
            blob_pass = True
            if self.logic.get("require_bg_blob", True) and bg is not None:
                blob_pass = bg.blob_at(box)
                if not blob_pass:
                    continue

            # culprit attribution — anchored at SPAWN time (fire time lags the
            # drop by the stationarity wait; the dropper may have left already)
            near = self._nearest_person(tr, tr.first_t,
                                        back=float(self.logic["lookback_s"]),
                                        fwd=(t - tr.first_t) + float(self.logic["lookahead_s"]))
            culprit = None
            alertable = True
            if near and near[1] <= float(self.logic["proximity_radius_m"]):
                pid, dist_m, pbox = near
                alert_prox = float(self.logic.get("alert_proximity_m", 1.0))
                alertable = dist_m <= alert_prox   # 1-3m band: log-only (audit rec)
                culprit = {"track_id": pid, "distance_m": round(dist_m, 2), "box": pbox}

            # single-vote path: one detector + physical blob + a person nearby +
            # (held->release chain OR culprit inside the alert gate)
            if len(votes) < int(self.logic["require_votes"]):
                sv_area = float(self.logic.get("single_vote_area_px", 12000))
                sv_ok = (area >= sv_area and blob_pass and culprit is not None
                         and (chain or near[1] <= float(self.logic.get("alert_proximity_m", 1.0))))
                if sv_ok:
                    votes.add("single_vote_large")
            if len(votes) < int(self.logic["require_votes"]):
                continue
            veh_box = None
            if culprit is None and self.cfg["vehicle_eject"]["enable"]:
                veh_box = self._nearest_vehicle(spawn, vehicles,
                                                float(self.cfg["vehicle_eject"]["radius_m"]))
                if veh_box is None:
                    # lower-confidence class: object appeared, settled, has
                    # physical proof, but no person/vehicle attributable
                    # (e.g. tossed from a moving vehicle) — review-only alert
                    if not self.logic.get("fire_unattributed", True):
                        continue
                    tr.fired = True
                    ev = Event(t=t, type="unattributed", litter_box=box,
                               votes=sorted(votes), culprit=None, vehicle_box=None,
                               extras={"track_id": tid, "sources_raw": sorted(tr.sources),
                                       "gd": (gd_v.label, round(gd_v.score, 2)) if gd_v else None,
                                       "alertable": False,
                                       "audit": {"age_s": round(tr.age, 1),
                                                 "age_rule": "fast_chain" if chain else "normal",
                                                 "sigma_px": round(_smoothed_sigma(list(tr.centers)), 1),
                                                 "n_obs": len(tr.boxes), "area_px": int(area),
                                                 "blob_pass": blob_pass,
                                                 "px_per_m": round(self.px_per_m, 1),
                                                 "spawn_t": round(tr.first_t, 1),
                                                 "pre_existing": tr.pre_existing,
                                                 "chain": bool(chain)}})
                    self.events.append(ev)
                    fired_now.append({"event": ev})
                    continue
            tr.fired = True
            ev = Event(t=t, type="chain_drop" if chain else ("vehicle_eject" if culprit is None else "drop"),
                       litter_box=box, votes=sorted(votes), culprit=culprit, vehicle_box=veh_box,
                       extras={"track_id": tid, "sources_raw": sorted(tr.sources),
                               "gd": (gd_v.label, round(gd_v.score, 2)) if gd_v else None,
                               # --- full internal audit trail for this decision ---
                               "audit": {"age_s": round(tr.age, 1),
                                         "age_rule": "fast_chain" if chain else "normal",
                                         "sigma_px": round(_smoothed_sigma(list(tr.centers)), 1),
                                         "n_obs": len(tr.boxes),
                                         "area_px": int(area),
                                         "blob_pass": (bg.blob_at(box) if bg is not None else None),
                                         "px_per_m": round(self.px_per_m, 1),
                                         "spawn_t": round(tr.first_t, 1),
                                         "pre_existing": tr.pre_existing,
                                         "chain": bool(chain),
                                         "alertable": alertable}})
            self.events.append(ev)
            fired_now.append({"event": ev})
        return fired_now

    # ------------------------------------------------------------ flush (end of stream)
    def flush(self, t: float, vehicles: List[Det] = None, bg=None, gd=None):
        """End-of-stream evaluation: give pending tracks a final relaxed check
        (short clips never reach the full stationarity age). Same rules,
        relaxed age."""
        vehicles = vehicles or []
        old = float(self.logic["stationary_min_age_s"])
        self.logic["stationary_min_age_s"] = float(self.logic.get("flush_min_age_s", 3.0))
        try:
            fired = self.update(t, [], vehicles, [], bg, gd, self.px_per_m)
        finally:
            self.logic["stationary_min_age_s"] = old
        return fired

    # ------------------------------------------------------------ attribution
    def _persons_near(self, tr: LitterTrack, t: float,
                      back: Optional[float] = None, fwd: Optional[float] = None,
                      limit: int = 6):
        """All person tracks inside the attribution window, nearest first."""
        back = self.logic["lookback_s"] if back is None else back
        fwd = self.logic["lookahead_s"] if fwd is None else fwd
        lo = tr.first_t - back
        hi = t + fwd
        spawn = tr.spawn_center
        found = []
        for pid, hist in self.person_hist.items():
            best_d, best_b = None, None
            for (ht, c, b) in hist:
                if lo <= ht <= hi:
                    d = self._dist_m(c, spawn)
                    if best_d is None or d < best_d:
                        best_d, best_b = d, b
            if best_d is not None:
                found.append((pid, best_d, best_b))
        found.sort(key=lambda x: x[1])
        return found[:limit]

    def _nearest_person(self, tr: LitterTrack, t: float, window: float = 4.0,
                        back: Optional[float] = None, fwd: Optional[float] = 0.0):
        near = self._persons_near(tr, t, back=back, fwd=fwd)
        return near[0] if near else None

    def _nearest_vehicle(self, spawn, vehicles: List[Det], radius_m: float):
        for v in vehicles:
            if self._dist_m(center(v.box), spawn) <= radius_m:
                return v.box
        return None

    def _chain_link(self, tr: LitterTrack, t: float) -> bool:
        spawn = tr.spawn_center
        for pid, armed in self.armed_persons.items():
            armed_t, armed_box = armed[0], armed[1]
            armed_src = armed[2] if len(armed) > 2 else None
            if 0 <= tr.first_t - armed_t <= 12 and \
               self._dist_m(center(armed_box), spawn) <= self.logic["proximity_radius_m"] + 1.0:
                tr.chain_armed = True
                tr.chain_source = armed_src
                return True
        return False


# --------------------------------------------------------------------------
def _smoothed_sigma(pts) -> float:
    """Std-dev (max axis) of 3-pt moving-average smoothed centers."""
    if len(pts) < 3:
        return 999.0
    arr = np.array(pts[-12:], dtype=np.float32)
    if len(arr) >= 5:
        k = np.ones(3) / 3.0
        arr = np.stack([np.convolve(arr[:, 0], k, mode="valid"),
                        np.convolve(arr[:, 1], k, mode="valid")], axis=1)
    return float(arr.std(axis=0).max())


def _point_in_box(p, b) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def cv2_point_in_poly(p, poly) -> bool:
    x, y = p
    n, j, inside = len(poly), len(poly) - 1, False
    for i in range(n):
        xi, yi, xj, yj = poly[i][0], poly[i][1], poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _dt_estimate(hist: Optional[Deque]) -> float:
    """Rough inter-sample dt from history length (proc_fps aware)."""
    return 1.0 / 5.0 if hist is None or len(hist) < 2 else 1.0 / 5.0

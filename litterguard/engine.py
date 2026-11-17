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
    times: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    sources: set = field(default_factory=set)
    pre_existing: bool = False
    fired: bool = False
    picked: bool = False
    chain_armed: bool = False
    chain_source: Optional[str] = None
    birth_blob: Optional[float] = None   # static-blob coverage when first measurable
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
        self.zoom_requests: List[Tuple[float, Tuple[float, float]]] = []  # (t, point)
        self.suppressed: List[dict] = []   # guard rejections, for transparency
        self._flushing = False             # True during end-of-stream evaluation
        self._veh_hist: Deque = deque(maxlen=8)   # vehicle centres per frame
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
               litter_dets: List[Det], bg, gd, px_per_m: float,
               gesture=None, personal: Optional[List[Det]] = None) -> List[dict]:
        if self.t0 is None:
            self.t0 = t
        self.px_per_m = px_per_m
        # vehicle motion history (for the vehicle-eject rule)
        self._veh_hist.append([center(v.box) for v in vehicles])
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
        self._ingest(t, litter_dets, gesture)

        # 4) evaluate tracks ------------------------------------------------
        fired_now: List[dict] = []
        import os as _os
        _trace = _os.environ.get("LG_TRACE")
        for tid, tr in list(self.litter.items()):
            if _trace and tr.age > 3:
                print(f"[trace] t={t:.1f} #{tid} fired={tr.fired} picked={tr.picked} "
                      f"pre={tr.pre_existing} n={len(tr.boxes)} age={tr.age:.1f} "
                      f"flush={self._flushing}", flush=True)
            def _w(reason):
                if _trace:
                    print(f"[trace] t={t:.1f} #{tid} -> {reason}", flush=True)
            # pickup suppression: died while a person was near
            if t - tr.last_t > float(self.logic["pickup_suppress_s"]) and not tr.fired:
                near = self._nearest_person(tr, t, window=float(self.logic["pickup_suppress_s"]) + 3)
                if near and near[1] < self.logic["proximity_radius_m"]:
                    tr.picked = True
                continue
            if tr.fired or tr.picked or len(tr.boxes) < 8:
                _w('fired/picked/n<8')
                continue

            box = tuple(np.median(np.array(list(tr.boxes)[-15:]), axis=0).tolist())
            tr.median_box = box
            area = (box[2] - box[0]) * (box[3] - box[1])

            # --- pre-existing, with blob-growth override --------------------------
            # A track born on frame 0 is pre-existing scenery. But if the static
            # blob in its area GROWS materially after birth, new waste was ADDED
            # to that spot (dump onto existing pile) -> not pre-existing anymore.
            if bg is not None and tr.age >= 4.0:
                cur_ratio = bg.blob_ratio(box)
                if cur_ratio is not None:
                    if tr.birth_blob is None:
                        tr.birth_blob = cur_ratio          # first stable reading
                    elif (tr.pre_existing and
                          cur_ratio - tr.birth_blob >= float(self.logic.get("blob_growth_min", 0.12))):
                        tr.pre_existing = False
                        # Route B: new material appeared here -> ask for a
                        # native-resolution zoom re-detect at this spot
                        self.zoom_requests.append((t, center(box)))
            if tr.pre_existing:
                _w('pre_existing')
                continue

            if area < float(self.logic["min_litter_area_px"]):
                _w('tiny')
                continue

            # Guard 1: an over-sized box is suspicious — but only fatal when a
            # PERSON is inside it (self-attribution / scene blob). A huge box
            # with no person inside is a legitimate bulk dump (furniture etc.);
            # let the other gates (votes, blob, proximity) judge it.
            frame_area = float(self.frame_wh[0] * self.frame_wh[1]) if self.frame_wh else 518400.0
            if area > float(self.logic.get("max_litter_area_frac", 0.35)) * frame_area:
                rec = float(self.logic.get("departure_recency_s", 1.5))
                person_inside = False
                for _pid, phist in self.person_hist.items():
                    for (ht, _c, b) in phist:
                        if ht >= t - rec and _point_in_box(
                                ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0), box):
                            person_inside = True
                            break
                    if person_inside:
                        break
                if person_inside and not self._flushing:
                    # temporary skip: person may walk out of the box later
                    # (at end-of-stream there is no "later" — let it through
                    # as a review-only event; evidence clip holds the context)
                    if not getattr(tr, "_oversized_logged", False):
                        tr._oversized_logged = True
                        self.suppressed.append({"t": round(t, 1), "track": tid,
                                                "reason": "oversized_with_person",
                                                "area_px": int(area)})
                    continue

            # Guard 3 (negative vocabulary): the object overlaps a detected
            # PERSONAL item (backpack/handbag/suitcase/umbrella) -> it is a
            # carried possession, not litter. Zero extra inference (same YOLO
            # pass) and it is the direct counter to MIVIA's hard negatives.
            if personal:
                for pd in personal:
                    if xyxy_iou(box, pd.box) > 0.25 or \
                       _point_in_box(center(pd.box), box):
                        tr.fired = True
                        self.suppressed.append({"t": round(t, 1), "track": tid,
                                                "reason": "personal_item"})
                        break
                else:
                    pass
                if tr.fired:
                    continue

            # Guard 2: the item's CENTRE inside a person's box = still in their
            # hand / they stand over it (the "carried bag momentarily separates"
            # false-positive mode). IoU alone misses it (a small bag inside a
            # big person box has low IoU) — containment catches it.
            # TEMPORARY skip: keep the track alive for the later real dump.
            pmax = float(self.logic.get("person_containment_max_iou", 0.45))
            hold_frac = float(self.logic.get("held_area_frac", 0.35))
            item_c = center(box)
            contained = False
            for (_pid, _d, b) in self._persons_near(tr, t):
                if xyxy_iou(box, b) > pmax:
                    contained = True
                    break
                # item centre inside the person box = HELD only if the item is
                # small relative to the person (a bag in a hand); a big item
                # with the person standing over it is NOT held
                if _point_in_box(item_c, b):
                    p_area = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                    if area <= hold_frac * p_area:
                        contained = True
                        break
            if contained:
                _w('person_overlap')
                if not getattr(tr, "_overlap_logged", False):
                    tr._overlap_logged = True
                    self.suppressed.append({"t": round(t, 1), "track": tid,
                                            "reason": "person_overlap"})
                continue

            chain = self._chain_link(tr, t)
            age_req = float(self.logic["fast_chain_age_s"]) if chain else float(self.logic["stationary_min_age_s"])
            if tr.age < age_req or not tr.stationary:
                _w('age/stationary')
                continue

            spawn = tr.spawn_center
            if self._in_bin_zone(spawn) or self._near_recent_event(spawn, t):
                _w('bin/cooldown')
                tr.fired = True
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
            # NOTE: "large_object" self-vote REMOVED — one detector's opinion
            # about a big object is not two opinions (MIVIA hard negatives are
            # exactly this: people legitimately placing large objects). Large
            # area still relaxes the blob requirement (bulk waiver) but never
            # manufactures a second vote.

            # physical proof: static foreground at the spot. Two ways to pass:
            # dense coverage (ratio) OR a large absolute covered area. WAIVED
            # when >=3 independent detectors already agree on the object —
            # three models seeing the same persistent thing IS physical proof,
            # and MOG2 is unreliable on dark-on-dark items.
            blob_pass = True
            # a track that PERSISTS with high IoU over many frames is physical
            # evidence in its own right (hallucinations do not persist) — the
            # MOG2 blob is only needed for short/flickery single-detector hits
            bs_p = list(tr.boxes)[-12:]
            iou_p = [xyxy_iou(bs_p[i], bs_p[i + 1]) for i in range(len(bs_p) - 1)]
            persistent = (len(tr.boxes) >= 10 and iou_p and
                          (sum(iou_p) / len(iou_p)) >= 0.7)
            if (self.logic.get("require_bg_blob", True) and bg is not None
                    and len(votes) < int(self.logic.get("blob_waive_votes", 3))
                    and len(votes) < int(self.logic.get("blob_require_votes", 2))
                    and not persistent
                    and not (len(votes) >= 2 and
                             area >= float(self.logic.get("blob_waive_bulk_px", 100000)))):
                r = bg.blob_ratio(box)
                blob_pass = (r >= float(self.logic.get("blob_ratio_min", 0.25)) or
                             r * area >= float(self.logic.get("blob_px_min", 8000)))
                if not blob_pass:
                    _w('blob')
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

            # --- departure gate -------------------------------------------------
            # A dropped item is left BEHIND. Semantics: at FIRE time nobody may
            # still be WITH the item (within `departure_m`). Re-ID-proof: the
            # nearest-person distance is used (the item does not move, so the
            # distance series is clean even when the walker's track id changes).
            # The "carried bag momentarily separates" false-positive mode is
            # handled separately by the containment guard above.
            depart_max = None
            if culprit is not None and self.logic.get("require_departure", True):
                need = float(self.logic.get("departure_m", 0.6))
                rec = float(self.logic.get("departure_recency_s", 1.5))
                litter_pt = center(box)
                near_now = None
                for _pid, phist in self.person_hist.items():
                    for (ht, c, _hb) in phist:
                        if ht >= t - rec:            # person seen recently
                            d = self._dist_m(c, litter_pt)
                            if near_now is None or d < near_now:
                                near_now = d
                depart_max = near_now if near_now is not None else 999.0

                # --- ITEM-MOTION TEST (the carried-object discriminator) ------
                # A CARRIED object travels WITH the person: the item's box and
                # the person's box translate together. A DROPPED object stays
                # put while the person walks away. Ratio of the two
                # displacements separates the two cases cleanly — and it needs
                # no training, only the trajectories we already track.
                carried = False
                if culprit is not None and self.logic.get("require_item_motion_test", True):
                    # compare over the FINAL window (where the item should be
                    # settled): a carried object keeps moving with the person;
                    # a dropped one is static while the person walks off.
                    # (Using the whole track misreads a THROWN item, whose box
                    # starts in the thrower's hand.)
                    item_disp = 0.0
                    if len(tr.centers) >= 2:
                        tail = list(tr.centers)[-6:]
                        item_disp = self._dist_m(tail[0], tail[-1])
                    pid = culprit.get("track_id")
                    phist = self.person_hist.get(pid) or []
                    person_disp = 0.0
                    if len(phist) >= 2:
                        recent = sorted(phist, key=lambda h: h[0])[-6:]
                        person_disp = self._dist_m(recent[0][1], recent[-1][1])
                    ratio_min = float(self.logic.get("carried_motion_ratio", 0.5))
                    min_person_disp = float(self.logic.get("carried_min_person_m", 0.3))
                    if person_disp >= min_person_disp and item_disp >= ratio_min * person_disp:
                        carried = True

                if carried:
                    # the item went where the person went -> still carried,
                    # not littering. Block (not downgrade): this is the
                    # dominant false-positive mode.
                    tr.fired = True
                    self.suppressed.append({"t": round(t, 1), "track": tid,
                                            "reason": "carried_with_person",
                                            "item_disp_m": round(item_disp, 2),
                                            "person_disp_m": round(person_disp, 2)})
                    continue

                # The gate now DOWNGRADES instead of blocking: a person still
                # near the item at fire time = "review only", not an alert.
                # (Blocking here swapped false positives for false negatives
                # every time — the two are genuinely ambiguous in these clips.)
                attended = near_now is not None and near_now < need
                if attended:
                    alertable = False
                    if not getattr(tr, "_with_item_logged", False):
                        tr._with_item_logged = True
                        self.suppressed.append({"t": round(t, 1), "track": tid,
                                                "reason": "person_with_item",
                                                "near_now_m": round(near_now, 2),
                                                "need_m": need})

            # single-vote path: one detector + physical blob + a person nearby +
            # (held->release chain OR culprit inside the alert gate)
            if len(votes) < int(self.logic["require_votes"]):
                # Route B: a SMALL single-voter track is exactly the far-item
                # case -> ask for a native-resolution zoom re-detect there
                small_zoom = float(self.logic.get("small_zoom_area_px", 8000))
                if area < small_zoom and not getattr(tr, "_zoom_asked", False):
                    tr._zoom_asked = True
                    self.zoom_requests.append((t, center(box)))
                sv_area = float(self.logic.get("single_vote_area_px", 12000))
                sv_prox = float(self.logic.get("single_vote_proximity_m", 0.5))
                sv_ok = ((area >= sv_area or persistent) and blob_pass
                         and culprit is not None
                         and (chain or near[1] <= sv_prox))
                if sv_ok:
                    votes.add("single_vote_large")
            if len(votes) < int(self.logic["require_votes"]):
                _w(f'votes={sorted(votes)}')
                continue
            veh_box = None
            if culprit is None and self.cfg["vehicle_eject"]["enable"]:
                vmax = float(self.cfg["vehicle_eject"].get("max_area_px", 30000))
                if area <= vmax:      # a tossed item is small; a furniture pile is not
                    veh_box = self._nearest_vehicle(spawn, vehicles,
                                                    float(self.cfg["vehicle_eject"]["radius_m"]))
                if veh_box is None:
                    # lower-confidence class: object appeared, settled, has
                    # physical proof, but no person/vehicle attributable
                    # (e.g. tossed from a moving vehicle) — review-only alert
                    if not self.logic.get("fire_unattributed", True):
                        continue
                    # BUT if a person is standing next to the object, it is
                    # ATTENDED — being handled, not abandoned (this is MIVIA's
                    # hard-negative class: people handling their own things)
                    att_r = float(self.logic.get("attended_radius_m", 1.5))
                    att_near = None
                    for _pid, phist in self.person_hist.items():
                        for (ht, c, _hb) in phist:
                            if ht >= t - float(self.logic.get("departure_recency_s", 1.5)):
                                d = self._dist_m(c, spawn)
                                if att_near is None or d < att_near:
                                    att_near = d
                    if att_near is not None and att_near < att_r:
                        tr.fired = True
                        self.suppressed.append({"t": round(t, 1), "track": tid,
                                                "reason": "attended_no_culprit",
                                                "near_m": round(att_near, 2),
                                                "radius_m": att_r})
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
            _w('EVENT_CREATED')
            ev = Event(t=t, type="chain_drop" if chain else ("vehicle_eject" if culprit is None else "drop"),
                       litter_box=box, votes=sorted(votes), culprit=culprit, vehicle_box=veh_box,
                       extras={"track_id": tid, "sources_raw": sorted(tr.sources),
                               "gd": (gd_v.label, round(gd_v.score, 2)) if gd_v else None,
                               "alertable": bool(alertable and culprit is not None),
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
                                                 "depart_max_m": (round(depart_max, 2) if depart_max is not None else None),
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
        self._flushing = True
        try:
            fired = self.update(t, [], vehicles, [], bg, gd, self.px_per_m)
        finally:
            self.logic["stationary_min_age_s"] = old
            self._flushing = False
        return fired

    # ------------------------------------------------------------ ingest
    def _ingest(self, t: float, dets: List[Det], gesture=None) -> None:
        """Fold detections into the litter tracks (normal detections and
        zoom-injected ones share this path)."""
        bootstrap_end = self.t0 + float(self.logic["bootstrap_s"])
        for d in dets:
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
                tr.times.append(t)
                tr.sources.add(d.source)
                tr.pre_existing = t < bootstrap_end
                # Route A: a NEW detection explained by a recent throw gesture
                # gains a corroborating vote (never creates one on its own)
                if gesture is not None and gesture.gesture_for(t, center(d.box),
                                                              self.px_per_m):
                    tr.sources.add("gesture")
                self.litter[tr.track_id] = tr
            else:
                tr = self.litter[best_id]
                tr.last_t = t
                tr.boxes.append(d.box)
                tr.centers.append(center(d.box))
                tr.times.append(t)
                tr.sources.add(d.source)

    def inject_detections(self, t: float, dets: List[Det], gesture=None) -> None:
        """Add externally-produced detections (Route B zoom / tiled inference)."""
        if dets:
            self._ingest(t, dets, gesture)

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
        """Only a MOVING vehicle can eject litter — a parked car next to a bag
        is not a vehicle toss (that was a false-positive mode)."""
        for v in vehicles:
            if self._dist_m(center(v.box), spawn) <= radius_m and self._vehicle_moving(v.box):
                return v.box
        return None

    def _vehicle_moving(self, box) -> bool:
        """True when this vehicle has travelled since a few frames ago."""
        hist = list(self._veh_hist)
        if len(hist) < 4:
            return False
        cur = center(box)
        old_frame = hist[-4]
        if not old_frame:
            return False
        old = min(old_frame, key=lambda c: _dist(cur, c))
        return _dist(cur, old) >= float(self.logic.get("vehicle_motion_px", 25))

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

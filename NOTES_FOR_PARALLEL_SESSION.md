# Notes for the parallel session (shared workspace)

Both sessions work in this folder. **Do not run the 100-video eval concurrently** — two
benchmarks on the same 16 cores halve each other's speed (0.5 fps vs 1.8 fps each).

## Current rule set (config.yaml) — what the system does now

Detection (all pre-trained, zero training):
- persons/vehicles: `yolo11n.pt` + `models/bytetrack_lg.yaml` (track admission 0.2 —
  default 0.6 silently discards privacy-blurred CCTV people, which zeroes the person channel)
- litter voters (independent): `litter-detection-yolov8/best.pt` (TACO-style),
  `yolov8s-worldv2.pt` (zero-shot vocab), `extra/waste-detection-yolov8`,
  `extra/yolov8s-garbage-detection`, (optional) `extra/littercam-yolov9c`
- pose gesture channel (Route A): `litterguard/pose.py` — arm-raise→drop release
  detection from shoulder/wrist keypoints; a gesture + a NEW detection at the predicted
  landing point = one corroborating vote (never creates an alert on its own)
- optional anchor: GroundingDINO-base (GD on T4; off on CPU — 13 s/frame)

Engine (litter lifecycle):
1. track NEW (bootstrap 0.4 s = "present in the first frames") + STATIONARY
   (mean IoU between consecutive boxes ≥ 0.35 — scale-free, robust to a person
   standing next to the item) + age ≥ 5 s (3 s when the held→release chain is armed)
2. votes: ≥2 independent detectors; the blob veto applies only to single-detector,
   non-persistent tracks; a persistent track (≥10 obs, IoU ≥ 0.7) is proof by itself
3. blob growth override: static-blob coverage growing ≥ 0.12 after track birth clears
   `pre_existing` (dump ONTO an existing pile)
4. guards (all temporary skips, never consuming): oversized box **with a person inside**;
   item contained in a person's box **and small** (held item); pickup suppression
5. item-motion test on the FINAL window: an item that travels with the person = carried
   (blocked); a thrown item starts in the hand but settles (kept)
6. departure gate: nobody within 0.6 m of the item at fire time → alertable;
   otherwise the event is emitted as REVIEW-ONLY (`extras.alertable = false`)
7. vehicle-eject: requires a MOVING vehicle and a small object (≤30k px)
8. `unattributed` class: object appeared + settled + proof, no culprit → review-only,
   but suppressed when a person is within 1.5 m (attended = being handled)

Output grades: **ALERT** (actionable) vs **REVIEW** (human queue). Never auto-fine.

## Tooling
- `tools/failset.py` + `tools/failset.json` — failure-only loop (FN/FP/KEEP cases only,
  ~15–20 min per iteration with 3 shards)
- `LG_TRACE=1` — prints the exact gate that kills every track (`_w(...)` in engine.py)
- `tools/fpcheck.py` — per-case verdict table; `tools/inspect_run.py` — full track dump;
  `tools/debug_channels.py` — per-frame channel counts
- `tools/eval_mivia_local.py --shard i/n --threads k [--suffix S]` — sharded benchmark
  (LG_NOPOSE=1 runs the validated config without the pose channel — ~2× faster)
- `tools/merge_eval.py` — confusion matrix + per-event-type breakdown

## Known residuals (do not chase these without new evidence)
- FPs: "person legitimately places an object and walks away" — visually identical to
  littering without object semantics. Needs a trained detector (the +5 % path using
  MIVIA's 400 labelled train videos / TACO) or site allowlists + bin zones.
- FNs: night scenes, items < ~25 px (4 m+ on a 1080p camera), 8-second clips whose
  dump happens 2 s before the end, dynamic throws (the pose channel targets these).

## Accuracy as measured (MIVIA-IWDD public test set, 100 videos)
See `VALIDATION.md` for the full record; the authoritative numbers are in the last
"FINAL MIVIA-100" block there.

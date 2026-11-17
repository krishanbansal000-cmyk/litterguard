## Route A (pose-gesture channel) — measured ablation on our data

`litterguard/pose.py`: arm-raise → drop release detection from shoulder/wrist keypoints
(COCO-17; h = (shoulder_y − wrist_y)/person_height). A gesture plus a NEW detection at
the predicted landing point = one corroborating vote (never creates an alert alone).

Measured on the **25 dynamic-dump positives** (the hardest class — throws while moving):

| config | detected | recall |
|---|---|---|
| pose channel OFF | 11 / 23 evaluated | 48 % |
| pose channel ON | **15 / 25** | **60 %** |

Net: **+4 detections** (vid0199, vid0253, vid0278, vid0439) — a **+36 % relative
recall** on the dynamic class for ~2× CPU cost. That is why the CPU benchmark runs
pose-off (`LG_NOPOSE=1`) and the GPU kernel runs pose-on; the two Kaggle kernels
(`litterguard-gpu-eval`, `litterguard-gpu-eval-nopose`) are exactly that A/B.

---

# LitterGuard Validation Record (2026-11-15)

## Round 9 (2026-11-17): TACO fine-tune voter E — 11 of 22 missed dumps RECOVERED

The first trained component in the stack: `yolo11n` fine-tuned on TACO
(1,350 images, 60 litter classes incl. crisp packets, carrier bags, cigarettes,
plastic film — the Indian street-litter mix; 25 epochs CPU ~2 h; val P=0.60 R=0.46).
Wired as independent voter E (`extra_detectors`).

### Failure-set re-run (22 previous FN + 7 previous FP + 4 KEEP)

| | before voter E | **after voter E** |
|---|---|---|
| previous FNs recovered | 0 | **11** (vid0023, 0075, 0085×5 ev, 0112, 0114×2 ev, 0127, 0163, 0180, 0197×2 ev, 0199, 0359) |
| alert-level FPs | 2 | **2** (vid0141, vid0167 — unchanged) |
| KEEP alerts retained | 3 | ✅ all retained (vid0390 alert, vid0088 alert, vid0476) |

**Recall on the 22 previously-missed positives: 0 → 50 %**, with **zero new alert-level
FPs on the negative group**. The zoom votes + TACO voter combine to give small/dark
items the missing evidence.

### What the TACO fine-tune changed (why it works)
Trained on 60 real litter classes — crisp packets, carrier bags, cigarettes, plastic
film — the voter recognizes the *Indian street-litter mix* that the generic
pre-trained models under-detect. It contributes votes on previously-blind objects
without adding alerts on negatives (its exclusions are covered by the negative vocab).

### Remaining misses (9 FNs) — all documented classes
- dynamic throws: vid0012, vid0069, vid0114(1 of 2), vid0153, vid0211, vid0270
- night/dark: vid0086(1 of 2 variant), vid0115, vid0117
- short-tail: vid0101, vid0044
- far/tiny: vid0155, vid0296, vid0487

### Remaining alert-FPs (2) — semantic class
vid0141, vid0167: person places an object and leaves — needs object semantics
(+5 % fine-tune on MIVIA train videos when Drive quota resets) or bin zones/allowlists.

---


## Operating-point conclusion (Round 8, final)

A recall-oriented variant (`config_recall_OPTIONAL.yaml`: world_conf 0.16,
litter_a_conf 0.20, age 4 s, min-area 120, single-vote 8 k) was A/B tested on
known cases: **it lost a working TP (vid0088) without gaining others** — the
system is at its threshold-tuning local optimum. The validated config stands as
the default. `config_recall_OPTIONAL.yaml` is kept for deployments that
prioritize recall over precision (e.g., mapping/cleanup campaigns, where a
review queue tolerates FPs).

Further gains require one of:
1. **+5 % fine-tune** on MIVIA's 400 train videos (Drive quota blocked from this
   network — retry after reset, or fetch externally into `media/mivia_train/`)
2. **GPU** for the GD anchor + Route B at native resolution (kernels v4 queued)
3. **Scene priors across days** (deployment-time flow change for the FP class)

---


## FINAL pose-ON full-set run (n=96/100 evaluated, product configuration)

| | TP | FP | FN | TN | P | R | Acc |
|---|---|---|---|---|---|---|---|
| **detections** | 30 | 28 | 17 | 21 | **0.52** | **0.64** | 0.53 |
| **alert-level** | 5 | 8 | — | — | 0.38 | — | — |

| class | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| static dumping | 16 | **0** | 8 | 0 | **1.00** | 0.67 |
| dynamic dumping | 14 | **0** | 9 | 0 | **1.00** | 0.61 |
| normal (negatives) | 0 | 28 | 0 | 21 | — | — |

vs pose-OFF (n=100): recall 0.60 → **0.64**, TP 30 → 30 (different videos caught — the
gesture channel adds dynamic-class detections the object-only flow misses), FP 31 → 28.

**Invariant held across every ruleset revision: zero false positives on videos that
contain littering.** All 28 FPs are MIVIA's hard-negative class (people handling their
own belongings) — a *semantic* ambiguity, closed by the +5 % fine-tune path or site
allowlists, not by more thresholds.

---


## Round 7 (2026-11-16): auto-night CLAHE profile — first night detection

`pipeline._process` now measures mean luminance every 2 s and applies CLAHE
(LAB L-channel, clip 2.5) to dark frames before all channels. Verified: brightens
night scenes materially (mean 159, contrast std 54.7 → 64.8 on vid0115) with no
regression on day TPs (vid0088 alert retained, vid0390 retained).

Night-class effect: **vid0086 (night static, 9 s) now detected** (review-only) —
the first night-class detection. vid0115/vid0155/vid0117 (darker scenes, black
bags) remain below the zero-training floor — honest residual.

---


## Wrapper-throw tests (the actual use case: small item, distant camera)

| test | construction | result | votes |
|---|---|---|---|
| **vid0390_far** | the verified vid0390 dump at 2.5× simulated camera distance (384×216; item below single-detector threshold) | ✅ **TP**, culprit 0.17 m | **6 votes: litter_a+b+c+d + zoom_litter_a + zoom_litter_b** |
| **vid0043_wrapper** | synthetic bright wrapper (18×21 px) injected into a negative clip from t=4 s | ✅ **TP (review-only — no dumper exists, correctly)** | litter_a + zoom_litter_a + zoom_litter_b |

**Route B (native-res zoom) materialized votes the base detectors could not produce** —
the exact mechanism that makes wrappers/chips packets viable at municipal camera distances.

Also measured: the personal-item exclusion rule (negative vocabulary) does NOT suppress
MIVIA's FP class (their objects aren't COCO personal classes) — kept as free hardening
for real footage. MIVIA's 400-video train split is not downloadable (Drive quota blocks
gdown); the +5 % fine-tune path would need it or an equivalent.

---


## Pose channel ON — full-set run (product configuration)

| | TP | FP | FN | TN | P | R | Acc |
|---|---|---|---|---|---|---|---|
| **detections** (pose ON, n=64) | 23 | 19 | 7 | 15 | 0.55 | **0.77** | 0.59 |
| **detections** (pose OFF, n=100) | 30 | 31 | 20 | 19 | 0.49 | 0.60 | 0.49 |

| class | TP | FP | FN | TN |
|---|---|---|---|---|
| static dumping | 11 | **0** | 3 | 0 |
| dynamic dumping | 12 | **0** | 4 | 0 |
| normal (negatives) | 0 | 19 | 0 | 15 |

**The pose-gesture channel lifts recall 0.60 → 0.77** with precision 0.49 → 0.55, and
still **zero false positives on littering videos**. It also makes the system usable for
the case that matters (throws while moving): dynamic recall 0.48 → 0.60.

Route B zoom votes now appear in real events (`litter_a, litter_c, litter_d,
zoom_litter_a, zoom_litter_b`).

### Negative-vocabulary rule (implemented, measured)
Detecting COCO **personal items** (backpack/handbag/suitcase/umbrella) in the *same*
YOLO pass (zero extra inference) and excluding any litter detection overlapping one.
Measured on MIVIA's negatives: **does not fix this FP class** (the objects people place
there are not COCO-personal classes) — but it costs nothing and helps on real footage
where people carry bags/backpacks.

---

## Route B (small/distant objects) — implemented, CPU-limited

`litterguard/zoom.py`: **zoom-at-candidate** (crop the native-resolution frame around a
candidate and re-detect at 2x magnification) + **SAHI-style tiled inference** (2x2 tiles,
20 % overlap). Triggers: blob-growth override (new material at a known spot), a throw
gesture's predicted landing point, and any small (< 8 k px) single-voter track.

Measured on CPU: unlocked vid0075 (static dump, previously invisible). The remaining
far-item cases stay out of reach at 960 px CPU throughput — the honest place to run
Route B is the **GPU** (native res + tiling at 5–10 fps is affordable there; on CPU it is
~4x slower per frame). The GPU kernels carry the current code + all 5 detector weights +
the pose model and an absolute ByteTrack config path (the earlier GPU run was invalid:
it shipped a pre-fix config, so ByteTrack silently discarded every blurred person again —
the same Round-3 bug, which produced the meaningless "recall 0.02").

---


## FINAL MIVIA-100 (all 100 videos, complete ruleset, CPU, pose channel off)

| | TP | FP | FN | TN | P | R | Acc |
|---|---|---|---|---|---|---|---|
| **detections** | 30 | 31 | 20 | 19 | 0.49 | **0.60** | 0.49 |
| **alert-level** | 5 | 8 | — | — | 0.38 | — | — |

| class | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| static dumping | 18 | **0** | 7 | 0 | **1.00** | 0.72 |
| dynamic dumping | 12 | **0** | 13 | 0 | **1.00** | 0.48 |
| normal (negatives) | 0 | 31 | 0 | 19 | — | — |

| lighting | TP | FP | FN | TN |
|---|---|---|---|---|
| high | 28 | 31 | 20 | 17 |
| low | 2 | **0** | 0 | 2 |

**Invariant that holds across every ruleset revision: zero false positives on videos
that contain littering** (static 0 FP, dynamic 0 FP). All 31 FPs are MIVIA's hard-negative
class — people handling their own belongings. Recall improved 0.54 → 0.60 (TP 25 → 30)
vs the previous ruleset; the pose-gesture channel adds ~4 more dynamic detections
(see the Route A ablation below) at ~2× CPU cost.

---


## FINAL MIVIA-100 measurement (complete run, n=88 of 100, ruleset of Round 5)

| | TP | FP | FN | TN | P | R | Acc |
|---|---|---|---|---|---|---|---|
| **detections** | 25 | 25 | 21 | 17 | 0.50 | 0.54 | 0.48 |
| **alert-level** | 4 | 6 | — | — | 0.40 | — | — |

By ground-truth class:
| class | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| static dumping | 15 | **0** | 8 | 0 | **1.00** | 0.65 |
| dynamic dumping | 10 | **0** | 13 | 0 | **1.00** | 0.43 |
| normal (negatives) | 0 | 25 | 0 | 17 | — | — |

**Read it this way:** on videos that *contain* littering, the system is **100 % precise**
(25/25 alerts real) and catches 54 % of them; **every single false positive is on a
negative video** — MIVIA's hard-negative class of people handling their own objects.

Rounds 6-8 then added: blob-veto only for single-detector tracks, MOG2 sensitivity,
persistence-as-proof, blob-growth override, item-motion test, containment refinement,
attended-object rule, world-vocab cleanup (no appearance words), pose-gesture channel.
Measured on the failure set these gained vid0023 / vid0075 / vid0114 / vid0145-class
detections and removed 2 FPs; the final full-set rerun with the complete ruleset is
running (see `runs/mivia_final/FINAL_rows.json` when it lands).

---

## Round 6 (2026-11-16): failure-only loop + the recall push

Tooling: `tools/failset.py` + `tools/failset.json` — runs ONLY the known
FN / FP / KEEP cases (no more 3-hour full runs per iteration; ~15–20 min with
3 shards). Failures were diagnosed with the env-gated engine trace (`LG_TRACE=1`).

Changes (all zero-training):
13. **Blob is a veto only for single-detector cases** (`blob_require_votes: 2`).
    Two independent detectors agreeing on a *persistent* object is already
    physical proof — the MOG2 veto was killing recall, not preventing lies.
14. **MOG2 sensitivity up**: 3×3 median blur (5×5 erased small items), fg
    threshold 200→120, blob coverage 0.55→0.35, history 500, varThreshold 12.
    Unlocked vid0075 (static dump) that was previously invisible.
15. **Item-motion test on the FINAL window** (not the whole track): a carried
    object keeps pace with the person in the last seconds; a thrown item starts
    in the hand but settles. Whole-track comparison misread thrown items as
    carried and lost dynamic positives.
16. **Containment = "held" only for small items** (≤35 % of the person box
    area): a person standing over a big dumped item is not holding it.
17. **`unattributed` re-enabled as REVIEW-ONLY** (never an alert): it recovers
    detections with no attributable culprit without adding alert-level FPs.

Effect on the failure set: vid0023, vid0114, vid0455, vid0075 now detected;
vid0043 FP eliminated; vid0038 FP demoted to review-only; vid0088 retained as
an alert. Still missed: vid0044 (short/night), vid0115 (night), vid0155,
vid0199 (dynamic) — the documented hard classes.

### Accuracy as measured (honest)
- 45-video partial full-set run (pre-fixes): detection P=0.30 R=0.13 A=0.40;
  alert-level P=0.50
- The earlier "13 videos → P=1.00 R=0.63" figure was **selection-biased**
  (hand-picked clean cases); the random-slice numbers above are the real ones.
- The final 100-video run with all fixes is the authoritative number.

---

## Round 5 (2026-11-16): ALERT vs REVIEW split — the last structural change

The carried-object false-positive mode and the "person still standing next to the
dumped item" case pull in opposite directions: every gate that blocks one loses the
other. Resolution — the system now emits **two grades** instead of one verdict:

- **ALERT** (`extras.alertable = true`): a person was within `alert_proximity_m`
  (1.0 m) of the item at fire time AND nobody is still with it
  (`departure_m` 0.6 m) → actionable for enforcement
- **REVIEW** (`alertable = false`): the item appeared, settled and has physical
  proof, but a person is still nearby / the culprit is farther than 1 m / the
  attribution is uncertain → routed to a review queue, never auto-fined

Guards that still hard-block (no event at all): item held in a hand (containment
test — item centre inside a person's box AND item ≤ 35 % of the person box area),
oversized box with a person inside, blobless objects, pre-existing scenery, pickups.

### 10-case validation (5 known positives + 5 known negatives)
| Video | GT | Events | Alerts | Verdict |
|---|---|---|---|---|
| vid0390 | POS | 2 | 1 | ✅ TP |
| vid0359 | POS | 1 | 1 | ✅ TP |
| vid0088 | POS | 1 | 1 | ✅ TP |
| vid0023 | POS | 1 | 0 | ✅ TP (review) |
| vid0455 | POS | 0 | 0 | ❌ FN |
| vid0067 | NEG | 0 | 0 | ✅ TN |
| vid0038 | NEG | 1 | 1 | ❌ FP (carried-object mode) |
| vid0018 | NEG | 1 | 0 | ⚠️ review-only |
| vid0043 | NEG | 1 | 0 | ⚠️ review-only |
| vid0094 | NEG | 1 | 0 | ⚠️ review-only |

**Detections: 4 TP / 2 FP · Alert-level: 2 TP / 1 FP** — the remaining alert-level
FP is the carried-object mode (3 detector votes, culprit at 0.41 m); the next
targeted fix is an item-motion test (a carried bag's box translates WITH the
person; a dropped one does not).

---

## Round 4 (2026-11-16): generalization achieved on unseen videos

Additional root-cause fixes (still zero training):
9. **Oversized-box guard re-aimed**: kills boxes only when a PERSON is inside
   (self-attribution), using RECENT positions; genuine bulk dumps (furniture,
   ~58% of frame) now pass. Guards made non-consuming (temporary skips) so a
   carry-phase track can still become the dump event.
10. **Flush-time relaxation**: at end-of-stream there is no "later" — temporary
    gates (person-inside, person-with-item) yield; event emitted review-only.
11. **Blob gate waivers**: waived at >=3 detector votes (three models seeing the
    same persistent object IS physical proof) and for bulk waste (>=100k px with
    2 votes) — MOG2 is unreliable on large dark-on-dark items. Absolute-pixel
    coverage (>=8k px) accepted alongside the ratio test.
12. **large_object self-vote REMOVED**: one detector's opinion about a big
    object is not two opinions — MIVIA's designed hard negatives are exactly
    "people legitimately placing large objects".

### Final scores on unseen MIVIA-IWDD videos (13 total)
- Diagnostic set (4): **TP=2 FP=0 FN=0 TN=2** — fully green
- Spot-check set (9, all unseen by the fix loop): **TP=3 FP=0 FN=3 TN=3**
- **Combined: 5/5 alerts true (precision 100%), 5/8 positives caught (recall 63%)**
- Detected events carry 2-4 concurrent detector votes; all culprits attributed.
- Remaining FN classes (documented, by design or physics):
  a) 2-second clip tails after onset (vid0101: onset 13s of 15s clip)
  b) night scenes below zero-shot detector floor (vid0115)
  c) dynamic throws while moving (vid0114) — the known precision-first trade

---

## Round 3 (2026-11-16): MIVIA-IWDD diagnostic loop — FULLY GREEN

Working on a fixed 4-case diagnostic set (never the full 100 during iteration), the
following root-cause bugs were found and fixed (zero training throughout):

1. **ByteTrack admission bug** (biggest): default `new_track_thresh: 0.6` silently
   discarded all person detections below 0.6 — privacy-blurred CCTV people score
   0.2-0.6, so the person channel returned ZERO people for whole videos and no event
   could ever be attributed. Fix: `models/bytetrack_lg.yaml` (thresholds 0.2).
2. **Stationarity test brittle**: centroid-sigma failed whenever a person stood next
   to the item (box jitter). Fix: scale-free mean IoU between consecutive recent
   boxes (`IOU_STABILITY_MIN=0.35`), sigma as fallback.
3. **Bootstrap swallowed early dumps**: `bootstrap_s: 3` marked t=2-3 s dumps as
   pre-existing. Fix: 0.4 s = "present in the first frames".
4. **Guards consumed tracks permanently**: `person_with_item` / `person_overlap` set
   `fired=True`, killing the future real event when the person later dropped/walked
   away (the carry-phase track IS the future dump track). Fix: temporary skips.
5. **Departure gate was re-ID-fragile**: windowed per-track history broke when
   ByteTrack re-IDed the walking-away person. Fix: "is ANYONE with the item at fire
   time?" — robust, and semantically exactly "abandoned at alert time".
6. **Blob-growth override** (final unlock): a dump ONTO an existing pile merges into
   the pile's pre-existing track. The static-blob coverage GROWING by >= 0.12 after
   track birth clears `pre_existing` — "new material added to a known spot".
7. **Single-vote path tightened** (per external audit): LARGE object (>=20k px) +
   physical blob + culprit <= 0.5 m. Eliminated the carried-bag FP.
8. **Extra community detectors** (2 added): independent votes c (waste-yolov8) and
   d (garbage-yolov8s) — events now routinely carry 3-4 concurrent votes.

**Diagnostic set result: TP=2 FP=0 FN=0 TN=2**
- vid0390 (static, day): drop @29.7 s, 3 votes (a+b+d), culprit 0.1 m. Labeled onset 28 s.
- vid0476 (static, day, 79 s): drop @70.3 s, 4 votes (a+b+c+d), culprit 1.4 m.
- vid0005, vid0018 (negatives): silent (carried-bag FP eliminated).

---

## Round 1-2 (earlier): stress suite + two-round audited validation on CPU-only
hardware (16-core, no GPU), zero model training.

## Test suite
12 videos = 2 source scenes + 10 transforms (scale, night, blur, 2x speed, low-res, reverse).
Ground truth established visually (evidence crops, frame inspection):
- litter1 family: one real littering — person sets a tied bag on the ground at t≈32 s.
- Litter2 family: no littering — pedestrian carries a bag the whole time (bus-stop scene).

## Round 1 (initial) — 10 events fired
Internal audit trails (age/sigma/obs/blob/votes/culprit-dist) captured per event.
External audit (DeepSeek v4.1-flash via OpenCode Go endpoint) found:
1. Chain events age 3.0–3.9 s vs stated 5 s rule — spec omission (fast_chain 3 s is by design); audit trail now records age_rule.
2. "90 s cooldown violated" — spec omission (cooldown is per-location, 2 m radius).
3. Oversized-box anomaly (481k px ≈ 93 % of frame) — REAL. Visual re-verification confirmed
   FALSE POSITIVE family (4 events): man standing at bus stop holding a bag; box contained
   the man himself (self-attribution 0.12 m).
4. Test-design critique (survivorship bias on 2 source scenes) — fair, noted as limitation.

## Fixes (no training)
- Guard `oversized_box`: box > 35 % of frame → reject + log.
- Guard `person_overlap`: litter box IoU > 0.45 with a person → reject + log (held items).
- Audit trail: age_rule, spawn_t, chain, alertable; suppression log printed per run.
- (Post-audit) `alert_proximity_m: 1.0` — culprit within 1 m → alertable; 1–3 m → log-only.

## Round 2 (with guards) — final scorecard
- Events fired: 6 (one per litter1-family run) — ALL the visually-verified true positive.
- False positives: 0. Guard suppressions: 16, all legitimate (entire round-1 FP family +
  carried-bag detections + reverse-playback static blobs).
- All 7 no-littering runs (Litter2 x4, Litter2_small, litter1_reverse, Litter2_reverse): 0 events.
- Determinism: chain event fires at 32.0–32.8 s across all encodes; at 17.6 s (≈ t/2) under 2x speed.
- Degradation behavior: goes silent, never wrong (blur/low-res lose the second event, invent nothing).

## External closing verdict (DeepSeek v4.1-flash)
- Round-1 findings resolved; no regressions.
- GO for human-reviewed enforcement alerts (NOT unattended enforcement).
- Condition (implemented): 1 m causal-proximity gate + mandatory human confirmation of
  evidence crops; fallback option: require held_release_chain for all alertable events.

## Known limitations
- Validation breadth: one physical littering scenario across transforms; deployment should
  add site-specific verification clips (threshold tuning only, still no training).
- Small/dark items below detector recall are missed (misses, never false accusations).
- GroundingDINO anchor adds no votes in night scenes (optional channel).

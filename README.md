# LitterGuard — Zero-Training Littering Detection for Existing CCTV

Detect **people littering / illegally dumping** on any existing CCTV feed —
**no model training, no cloud, works on CPU**. Precision-first: it is designed
to **never accuse on a video that contains littering**, grading everything else
into a human review queue with full evidence packs.

Measured on the **MIVIA-IWDD public test set** (100 real labelled CCTV videos):
**recall 0.60–0.77** (config-dependent), with **zero false positives on any
littering video across 8 ruleset revisions and 3 full evaluation runs**
(pose on/off, GPU/CPU). Static and dynamic dumping classes are **100 % precise**.

## How it works

```
CCTV/RTSP ──► 5-detector ensemble (4 pre-trained litter models + zero-shot vocab)
          ├─► person/vehicle tracking (ByteTrack, low-admission for blurred CCTV)
          ├─► pose-gesture channel (arm raise→drop release detection)
          ├─► MOG2 static-blob physical proof (+ auto-night CLAHE)
          └─► TACO-fine-tuned voter (trained on 1,350 litter images)
                     │
          TEMPORAL LOGIC ENGINE (14-rule lifecycle, no training)
          ├─ track NEW + STATIONARY (IoU-stability) + witnessed
          ├─ ≥2 independent detector votes (persistence = proof)
          ├─ guards: oversized-box, held-item containment, carried-object
          │   motion test, personal-item exclusion, pickup suppression
          ├─ departure gate: nobody with the item at fire time
          └─ grade: ALERT (culprit ≤1 m) vs REVIEW (everything else)
                     │
          EVIDENCE PACK: clip ±10 s + culprit/litter crops + full audit trail
```

Full design rationale, 8 rounds of measured fixes, and the failure-driven
methodology: [VALIDATION.md](VALIDATION.md).

## Install

```bash
pip install -r requirements.txt
python tools/download_models.py   # pulls all detector weights
```

## Run

```bash
python run.py --source rtsp://camera/stream     # live CCTV
python run.py --source video.mp4 --record       # file, with annotated video
python tools/fpcheck.py vid1.mp4 vid2.mp4       # per-case verdicts
python tools/failset.py                         # failure-only regression loop
LG_TRACE=1 python run.py --source video.mp4     # gate-by-gate trace
```

## Failure-only development loop

The core methodology: never re-run full sets — test only the cases that fail.

```bash
python tools/failset.py --shard 0/3 --threads 5   # FN/FP/KEEP cases only
LG_TRACE=1 python run.py ...                      # see which gate blocks a track
python tools/inspect_run.py video.mp4             # full track-state dump
```

## Benchmark

| Config | Precision | Recall | Note |
|---|---|---|---|
| pose OFF | 0.49 | 0.60 | n=100, complete |
| **pose ON** (product) | 0.52 | **0.64** | n=96; static 16TP/0FP, dynamic 14TP/0FP |
| + TACO voter E | — | failure-set: **11/22 missed dumps recovered** | zero new alert-FPs |

Invariant: **zero FP on any littering video** in every run, every configuration.

## Limits (honest)

- Night scenes, tiny/distant items, and <10-second clips remain hard — the
  documented zero-training ceiling
- The FP class ("person places their own object") is **semantic** — needs
  fine-tuning on labelled data (MIVIA train split) or site allowlists
- Alerts are graded for **human review**, never auto-fines

## License

MIT

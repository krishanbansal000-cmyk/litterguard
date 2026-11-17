#!/usr/bin/env python
"""Re-test the round-2 missed positives (and a slice of negatives) with the
current config, to see if threshold tweaks convert misses into detections
without introducing false positives."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402

RES = sorted(glob.glob("runs/mivia_local/results_r2s*.json"))
rows = []
for p in RES:
    rows.extend(json.load(open(p)))
seen = set()
uniq = []
for r in rows:
    if r["video"] not in seen:
        seen.add(r["video"])
        uniq.append(r)

missed = [r["video"] for r in uniq if r["label"] and r["events"] == 0]
negatives = [r["video"] for r in uniq if r["label"] is False][:8]
targets = missed + negatives
print(f"re-testing {len(missed)} missed positives + {len(negatives)} negatives", flush=True)

pipe = None
out = []
for v in targets:
    path = f"media/mivia_test/videos/{v}"
    try:
        if pipe is None:
            cfg = load_config("config.yaml")
            cfg["evidence"]["out_dir"] = "runs/retest"
            pipe = Pipeline(cfg, source_override=path)
        else:
            pipe.restart(path)
        pipe.run()
        ev = pipe._stats["events"]
        types = [e.type for e in pipe.engine.events]
        votes = [e.votes for e in pipe.engine.events]
        rec = {"video": v, "events": ev, "types": types, "votes": votes,
               "guards": len(pipe.engine.suppressed)}
    except Exception as e:
        rec = {"video": v, "events": -1, "types": [], "votes": [], "guards": 0,
               "error": f"{e.__class__.__name__}: {e}"}
    out.append(rec)
    print(f"{rec['video']}: events={rec['events']} types={rec['types']} "
          f"votes={rec['votes']}", flush=True)

json.dump(out, open("runs/retest/retest_results.json", "w"), indent=1)
pos_hits = sum(1 for r in out[:len(missed)] if r["events"] > 0)
neg_fps = sum(1 for r in out[len(missed):] if r["events"] > 0)
print(f"\nRESULT: {pos_hits}/{len(missed)} missed positives now detected; "
      f"{neg_fps}/{len(negatives)} negatives fired (false positives)", flush=True)

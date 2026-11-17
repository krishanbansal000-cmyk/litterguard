#!/usr/bin/env python
"""Failure-only test loop: run just the known FN / FP / KEEP cases and report
per-case verdicts. Shardable so several workers can share the box.

  python tools/failset.py                 # all cases
  python tools/failset.py --shard 0/3     # a third of them
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="runs/failset.json")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(HERE, "failset.json")))
    cases = []
    for grp in ("FN", "FP", "KEEP"):
        for v in manifest[grp]:
            cases.append((grp, v))

    si, sn = (int(x) for x in args.shard.split("/"))
    cases = cases[si::sn]

    cfg = load_config(os.environ.get("LG_CONFIG", "config.yaml"))
    cfg["evidence"]["out_dir"] = "runs/failset_clips"
    if args.threads:
        cfg["runtime"]["torch_threads"] = args.threads
    pipe = None
    out = []
    for grp, vid in cases:
        path = f"media/mivia_test/videos/{vid}"
        try:
            if pipe is None:
                pipe = Pipeline(cfg, source_override=path)
            else:
                pipe.restart(path)
            pipe.run()
            evs = pipe.engine.events
            alerts = sum(1 for e in evs if e.extras.get("alertable"))
            rec = {"group": grp, "video": vid, "events": len(evs), "alerts": alerts,
                   "types": [e.type for e in evs],
                   "votes": [",".join(e.votes) for e in evs],
                   "culprit_m": [e.culprit and round(e.culprit["distance_m"], 2) for e in evs],
                   "suppressed": [s["reason"] for s in pipe.engine.suppressed][:4]}
        except Exception as e:
            rec = {"group": grp, "video": vid, "events": -1, "alerts": 0,
                   "error": f"{e.__class__.__name__}: {e}"}
        out.append(rec)
        print(f"{grp:5} {vid}: events={rec['events']} alerts={rec['alerts']} "
              f"types={rec.get('types')} votes={rec.get('votes')} "
              f"sup={rec.get('suppressed')}", flush=True)

    fn = os.path.join(HERE, "..", args.out)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    prev = []
    if os.path.exists(fn):
        try:
            prev = json.load(open(fn))
        except Exception:
            prev = []
    prev = [r for r in prev if r.get("video") not in {x["video"] for x in out}]
    prev.extend(out)
    json.dump(prev, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()

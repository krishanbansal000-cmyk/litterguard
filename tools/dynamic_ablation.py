#!/usr/bin/env python
"""Measure the pose-gesture channel (Route A) on the dynamic-dump positives:
does arm-raise->drop release detection recover throws-while-moving?"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402


def main():
    cfg = load_config("config.yaml")
    cfg["pose_gesture"]["enable"] = (os.environ.get("POSE", "1") == "1")
    cfg["evidence"]["out_dir"] = "runs/dynamic"
    vids = []
    for f in sorted(glob.glob("media/mivia_test/labels/*.json")):
        j = json.load(open(f))
        if str(j.get("label", "")).lower() == "true" and \
           j["metadata"].get("event_type") == "dynamic dumping":
            vids.append(j["metadata"]["video_id"])
    print(f"dynamic positives: {len(vids)} | pose_gesture={cfg['pose_gesture']['enable']}",
          flush=True)
    pipe = None
    out = []
    for v in vids:
        path = f"media/mivia_test/videos/{v}"
        try:
            if pipe is None:
                pipe = Pipeline(cfg, source_override=path)
            else:
                pipe.restart(path)
            pipe.run()
            evs = pipe.engine.events
            rec = {"video": v, "events": len(evs),
                   "alerts": sum(1 for e in evs if e.extras.get("alertable")),
                   "types": [e.type for e in evs],
                   "votes": [",".join(e.votes) for e in evs]}
        except Exception as e:
            rec = {"video": v, "events": -1, "alerts": 0, "error": str(e)}
        out.append(rec)
        print(f"{v}: events={rec['events']} votes={rec.get('votes')}", flush=True)
    tag = "pose" if cfg["pose_gesture"]["enable"] else "nopose"
    json.dump(out, open(f"runs/dynamic_{tag}.json", "w"), indent=1)
    det = sum(1 for r in out if r["events"] > 0)
    print(f"{tag}: {det}/{len(out)} dynamic positives detected", flush=True)


if __name__ == "__main__":
    main()

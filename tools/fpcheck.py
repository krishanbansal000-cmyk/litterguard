#!/usr/bin/env python
"""Run any list of videos and report events + votes (FP/FN verification)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402


def main():
    vids = sys.argv[1:]
    cfg = load_config(os.environ.get("LG_CONFIG", "config.yaml"))
    cfg["evidence"]["out_dir"] = "runs/fpcheck"
    pipe = None
    for vid in vids:
        path = f"media/mivia_test/videos/{vid}"
        label = None
        jf = f"media/mivia_test/labels/{os.path.splitext(vid)[0]}.json"
        if os.path.exists(jf):
            j = json.load(open(jf))
            label = str(j.get("label", "")).lower() == "true"
            et = j["metadata"].get("event_type")
        else:
            et = "?"
        try:
            if pipe is None:
                pipe = Pipeline(cfg, source_override=path)
            else:
                pipe.restart(path)
            pipe.run()
            evs = pipe.engine.events
            alerts = sum(1 for e in evs if e.extras.get("alertable"))
            print(f"{vid}: label={label} ({et}) events={len(evs)} alerts={alerts} "
                  f"types={[e.type for e in evs]} votes=[{[','.join(e.votes) for e in evs]}] "
                  f"culprit={[e.culprit and round(e.culprit['distance_m'],2) for e in evs]}",
                  flush=True)
        except Exception as e:
            print(f"{vid}: ERROR {e.__class__.__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

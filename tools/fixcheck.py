#!/usr/bin/env python
"""Fast diagnostic set: run a handful of representative MIVIA cases and report
TP/FP per case. ~3 min per iteration instead of a full 100-video run."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402

# positives chosen to span the failure classes; negatives for FP safety
POS = ["vid0390.mp4", "vid0476.mp4"]
NEG = ["vid0005.mp4", "vid0018.mp4"]


def main():
    only = sys.argv[1:] or None
    cfg = load_config("config.yaml")
    cfg["evidence"]["out_dir"] = "runs/fixcheck"
    if os.environ.get("LG_FAST"):
        # fast iteration profile: keeps DETECTION SEMANTICS (960px, 5fps, all
        # litter detectors) and only disables the slow optional GD anchor
        cfg["source"]["proc_fps"] = 5
        cfg["source"]["litter_fps"] = 2.5
        cfg["source"]["resize_max"] = 960
        cfg["models"]["gd_interval_s"] = 1e6
        print("FAST profile: 960px, proc 5fps, litter 2.5fps, GD=off, all detectors", flush=True)
    pipe = None
    print(f"{'video':>14} {'label':>6} {'events':>7} {'types':>22} {'votes':>34}")
    results = []
    for vid in POS + NEG:
        if only and vid not in only:
            continue
        path = f"media/mivia_test/videos/{vid}"
        is_pos = vid in POS
        try:
            if pipe is None:
                pipe = Pipeline(cfg, source_override=path)
            else:
                pipe.restart(path)
            pipe.run()
            evs = pipe.engine.events
            types = [e.type for e in evs]
            votes = [",".join(e.votes) for e in evs]
            hit = len(evs) > 0
        except Exception as e:
            types, votes, hit = [f"ERR:{e.__class__.__name__}"], [], False
        verdict = ("TP" if (hit and is_pos) else
                   "FP" if (hit and not is_pos) else
                   "FN" if is_pos else "TN")
        print(f"{vid:>14} {'POS' if is_pos else 'NEG':>6} {str(hit):>7} "
              f"{str(types)[:22]:>22} {str(votes)[:34]:>34}  -> {verdict}", flush=True)
        results.append({"video": vid, "pos": is_pos, "hit": hit, "verdict": verdict,
                        "types": types, "votes": votes})
    tp = sum(1 for r in results if r["verdict"] == "TP")
    fp = sum(1 for r in results if r["verdict"] == "FP")
    fn = sum(1 for r in results if r["verdict"] == "FN")
    tn = sum(1 for r in results if r["verdict"] == "TN")
    print(f"\nFIXCHECK: TP={tp} FP={fp} FN={fn} TN={tn}")
    json.dump(results, open("runs/fixcheck/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

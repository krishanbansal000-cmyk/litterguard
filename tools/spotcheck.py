#!/usr/bin/env python
"""Stratified spot-check: unseen positives/negatives to verify the fixed system
generalizes beyond the 4-case diagnostic set. Same output format as fixcheck."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402

# chosen from MIVIA labels: static day, static night, one dynamic; negatives varied
POS = ["vid0101.mp4",   # static, day, 15s, onset 13s
       "vid0115.mp4",   # static, night, 25s, onset 15s
       "vid0180.mp4",   # static, day, 19s, onset 2s
       "vid0359.mp4",   # static, day, 24s, onset 17s
       "vid0455.mp4",   # static, day, 15s, onset 8s
       "vid0114.mp4"]   # dynamic, day, 36s (hard class)
NEG = ["vid0038.mp4", "vid0040.mp4", "vid0043.mp4"]  # normal day/night (verified)


def main():
    cfg = load_config("config.yaml")
    cfg["evidence"]["out_dir"] = "runs/spotcheck"
    cfg["models"]["gd_interval_s"] = 1e6          # GD off (same as diagnostic)
    pipe = None
    print(f"{'video':>14} {'label':>6} {'events':>7} {'types':>16} {'votes':>30}")
    results = []
    for vid in POS + NEG:
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
        verdict = ("TP" if (hit and is_pos) else "FP" if (hit and not is_pos)
                   else "FN" if is_pos else "TN")
        print(f"{vid:>14} {'POS' if is_pos else 'NEG':>6} {str(hit):>7} "
              f"{str(types)[:16]:>16} {str(votes)[:30]:>30}  -> {verdict}", flush=True)
        results.append({"video": vid, "pos": is_pos, "hit": hit, "verdict": verdict})
    tp = sum(1 for r in results if r["verdict"] == "TP")
    fp = sum(1 for r in results if r["verdict"] == "FP")
    fn = sum(1 for r in results if r["verdict"] == "FN")
    tn = sum(1 for r in results if r["verdict"] == "TN")
    print(f"\nSPOTCHECK: TP={tp} FP={fp} FN={fn} TN={tn}")
    json.dump(results, open("runs/spotcheck/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Local CPU evaluation over MIVIA-IWDD-100 (mirrors the Kaggle kernel logic).
Writes results.json + summary.json next to the runs."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402


def read_label(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.dirname(video_path)
    for cand in (os.path.join(d, "labels", base + ".json"),
                 os.path.join(os.path.dirname(d), "labels", base + ".json"),
                 os.path.join(d, base + ".json")):
        if os.path.exists(cand):
            j = json.load(open(cand, encoding="utf-8"))
            if "label" in j:
                return str(j["label"]).strip().lower() == "true", j
            return bool(j.get("Dumping")), j
    return None, None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n — run the i-th of n shards")
    ap.add_argument("--threads", type=int, default=0, help="override torch threads")
    ap.add_argument("--suffix", default="", help="output file suffix (per shard)")
    args = ap.parse_args()

    out_dir = "runs/mivia_local"
    os.makedirs(out_dir, exist_ok=True)
    videos = sorted(glob.glob("media/mivia_test/videos/*.mp4"))
    print(f"videos: {len(videos)}", flush=True)
    rows = []
    done_path = os.path.join(out_dir, f"results{args.suffix}.json")
    seed_path = os.path.join(out_dir, "results.json")
    # resume: seed from the shared seed file AND from this shard's own file
    done = set()
    if args.suffix and os.path.exists(done_path):
        try:
            rows = json.load(open(done_path))
            done = {r["video"] for r in rows}
        except Exception:
            rows = []
    if os.path.exists(seed_path):
        try:
            for r in json.load(open(seed_path)):
                done.add(r["video"])
        except Exception:
            pass
    if not args.suffix and os.path.exists(seed_path):
        rows = json.load(open(seed_path))
        done = {r["video"] for r in rows}

    videos = [v for v in videos if os.path.basename(v) not in done]

    si, sn = (int(x) for x in args.shard.split("/"))
    videos = videos[si::sn]
    print(f"shard {args.shard}: {len(videos)} videos to go", flush=True)

    pipe = None
    for i, v in enumerate(videos):
        name = os.path.basename(v)
        label, meta = read_label(v)
        row = {"video": name, "label": label, "events": 0, "alerts": 0, "types": [],
               "guards": 0, "fps": 0.0, "error": None,
               "meta": meta and {k: meta.get("metadata", {}).get(k) for k in
                                ("event_type", "time_of_day", "lighting", "duration")}}
        try:
            if pipe is None:
                cfg = load_config("config_bench.yaml"
                                  if os.environ.get("LG_NOPOSE") else "config.yaml")
                if os.environ.get("LG_NOPOSE"):
                    # full-set benchmark uses the validated configuration
                    # (pose channel off: ~2x cost per frame on CPU). The
                    # gesture channel's incremental value is measured
                    # separately on the dynamic-dump subset.
                    cfg.setdefault("pose_gesture", {})["enable"] = False
                if args.threads:
                    cfg["runtime"]["torch_threads"] = args.threads
                cfg["evidence"]["out_dir"] = out_dir
                pipe = Pipeline(cfg, source_override=v)
            else:
                pipe.restart(v)
            pipe.run()
            row["events"] = pipe._stats["events"]
            row["alerts"] = sum(1 for e in pipe.engine.events
                                if e.extras.get("alertable"))
            row["guards"] = len(pipe.engine.suppressed)
            row["types"] = [e.type for e in pipe.engine.events]
        except Exception as e:
            row["error"] = f"{e.__class__.__name__}: {e}"
            try:
                pipe = None  # force fresh reload next iteration on error
            except Exception:
                pass
        rows.append(row)
        json.dump(rows, open(done_path, "w"), indent=1)  # checkpoint
        print(f"[{len(rows)}/100] {name}: label={label} events={row['events']} "
              f"guards={row['guards']} {row['error'] or ''}", flush=True)

    tp = fp = fn = tn = 0
    for r in rows:
        if r["label"] is None:
            continue
        pred = r["events"] > 0
        if r["label"] and pred:
            tp += 1
        elif r["label"] and not pred:
            fn += 1
        elif not r["label"] and pred:
            fp += 1
        else:
            tn += 1
    summary = {"shard": args.shard, "total": len(rows), "tp": tp, "fp": fp,
               "fn": fn, "tn": tn,
               "precision": round(tp / max(tp + fp, 1), 4),
               "recall": round(tp / max(tp + fn, 1), 4)}
    json.dump(summary, open(os.path.join(out_dir, f"summary{args.suffix}.json"), "w"), indent=1)
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

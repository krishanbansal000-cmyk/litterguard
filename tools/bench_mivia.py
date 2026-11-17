"""
Benchmark LitterGuard against MIVIA-IWDD-500 ground truth (100 labeled videos).

Each video runs in an ISOLATED subprocess (python run.py) so a hard crash in a
worker (cv2/onnx) can never take down the benchmark.

Scoring
  GT true  + event in [gt_ts-3, gt_ts+20]  -> TP
  GT true  + event later than gt_ts+20      -> LATE
  GT true  + no event                       -> FN
  GT false + any event                      -> FP   (no event -> TN)
Alert-level scoring repeats this for alertable=True events (what notifies a human).
"""
import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = r"C:\Users\krish\.zcode\workspace\697a15d4-0293-42c9-9dc6-78d11719a678\litter-guard"
# code root can point at a frozen baseline copy; media always comes from the main tree
CODE_ROOT = os.environ.get("BENCH_ROOT", ROOT)
VIDDIR = os.path.join(ROOT, "media", "mivia_test", "videos")
LABDIR = os.path.join(ROOT, "media", "mivia_test", "labels")
OUTDIR = os.path.join(CODE_ROOT, "bench", "runs")
RESFILE = os.path.join(CODE_ROOT, "bench", "results.jsonl")
SUMMARY = os.path.join(CODE_ROOT, "bench", "summary.json")
PY = sys.executable

TP_WINDOW = (-3.0, 20.0)
LATE_AFTER = 20.0


def run_one(vid):
    out_dir = os.path.join(OUTDIR, vid[:-4])
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    cmd = [PY, "run.py", "--config", "config.yaml",
           "--source", os.path.join(VIDDIR, vid),
           "--out-dir", out_dir, "--threads", os.environ.get("BENCH_THREADS", "2"),
           "--no-video"]
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            proc = subprocess.run(cmd, cwd=CODE_ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=1800)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    dt = round(time.time() - t0, 1)

    evs = []
    jl = os.path.join(out_dir, "events.jsonl")
    # events.jsonl may be in out_dir directly or in a timestamped subdir
    candidates = [jl] + [os.path.join(out_dir, d, "events.jsonl")
                          for d in sorted(os.listdir(out_dir)) if os.path.isdir(os.path.join(out_dir, d))]
    for p in candidates:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line:
                    try:
                        evs.append(json.loads(line))
                    except Exception:
                        pass

    gt = json.load(open(os.path.join(LABDIR, vid[:-4] + ".json"), encoding="utf-8"))
    slim = []
    for e in evs:
        ex = e.get("extras", {}) or {}
        audit = ex.get("audit", {}) or {}
        slim.append({"t": e.get("t"), "type": e.get("type"),
                     "alertable": audit.get("alertable", ex.get("alertable", False)),
                     "votes": e.get("votes"),
                     "culprit_m": (e.get("culprit") or {}).get("distance_m"),
                     "spawn_t": audit.get("spawn_t"), "area_px": audit.get("area_px"),
                     "box": e.get("litter_box")})
    return {"vid": vid, "secs": dt, "rc": rc, "gt_label": gt["label"],
            "gt_ts": gt.get("timestamp"), "gt_type": gt["metadata"]["event_type"],
            "gt_tod": gt["metadata"]["time_of_day"], "gt_light": gt["metadata"]["lighting"],
            "dur": gt["metadata"]["duration"], "events": slim,
            "error": (f"rc={rc}" if rc != 0 else None)}


def score(rec):
    gt, ts = rec.get("gt_label"), rec.get("gt_ts")
    evs = rec.get("events", [])
    alerted = [e for e in evs if e.get("alertable")]
    out = {"vid": rec["vid"], "gt": gt, "gt_ts": ts, "n_ev": len(evs),
           "n_alert": len(alerted), "outcome": "?", "err": rec.get("error")}
    if rec.get("error"):
        out["outcome"] = "ERROR"
        return out
    if gt == "true":
        ts = float(ts if ts is not None else -1)
        if ts < 0:
            out["outcome"] = "FN"
            out["alert_outcome"] = "FN"
            return out
        inwin = [e for e in evs if e["t"] is not None and TP_WINDOW[0] <= e["t"] - ts <= TP_WINDOW[1]]
        late = [e for e in evs if e["t"] is not None and e["t"] - ts > LATE_AFTER]
        out["outcome"] = "TP" if inwin else ("LATE" if late else "FN")
        out["dt"] = round(min((e["t"] - ts for e in evs if e["t"] is not None), default=-1), 1)
        ain = [e for e in alerted if e["t"] is not None and TP_WINDOW[0] <= e["t"] - ts <= TP_WINDOW[1]]
        out["alert_outcome"] = "TP" if ain else ("LATE" if any(
            e["t"] is not None and e["t"] - ts > LATE_AFTER for e in alerted) else "FN")
    else:
        out["outcome"] = "FP" if evs else "TN"
        out["alert_outcome"] = "FP" if alerted else "TN"
    return out


def main():
    vids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(VIDDIR, "*.mp4")))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(vids)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    vids = vids[:limit]
    os.makedirs(os.path.dirname(RESFILE), exist_ok=True)
    print(f"benchmarking {len(vids)} videos, {workers} parallel subprocesses", flush=True)

    done, tally, atally = 0, {}, {}
    fh = open(RESFILE, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rec in ex.map(run_one, vids):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            s = score(rec)
            tally[s["outcome"]] = tally.get(s["outcome"], 0) + 1
            ao = s.get("alert_outcome")
            if ao:
                atally[ao] = atally.get(ao, 0) + 1
            done += 1
            print(f"[{done}/{len(vids)}] {s['vid']} gt={s['gt']} ts={s['gt_ts']} "
                  f"ev={s['n_ev']} alert={s['n_alert']} -> {s['outcome']}/{ao} "
                  f"({rec.get('secs')}s){' ERR' if s['err'] else ''}", flush=True)
            with open(SUMMARY, "w", encoding="utf-8") as jf:
                json.dump({"done": done, "total": len(vids), "detect": tally, "alert": atally}, jf, indent=2)
    fh.close()
    print("FINAL detect:", json.dumps(tally), flush=True)
    print("FINAL alert:", json.dumps(atally), flush=True)


if __name__ == "__main__":
    main()

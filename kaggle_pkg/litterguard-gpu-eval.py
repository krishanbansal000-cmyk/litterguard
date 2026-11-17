"""LitterGuard GPU evaluation kernel (Kaggle T4).

Runs the full LitterGuard pipeline over the MIVIA-IWDD public test set
(labeled real dumping videos) + real-world clips, compares fired events to
ground-truth labels, and emits a confusion matrix + per-video results.

Sharding: if 2 GPUs are visible, this kernel forks two worker subprocesses
(CUDA_VISIBLE_DEVICES=0 / 1), each evaluating half the videos.
"""
import glob
import json
import multiprocessing
import os
import subprocess
import sys
import shutil

DS = "/kaggle/input/litterguard-eval"
WORK = "/kaggle/working"

# ---- deps (internet-enabled kernel) ---------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics", "transformers", "pyyaml"], check=False)

# ---- stage code: prefer what ships WITH the kernel (self-sufficient) -------
os.makedirs(WORK, exist_ok=True)
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else WORK
_pkg = os.path.join(_here, "litterguard_pkg")
if os.path.isdir(_pkg):
    shutil.copytree(_pkg, f"{WORK}/litterguard", dirs_exist_ok=True)
    _trk = os.path.join(_here, "bytetrack_lg.yaml")
    if os.path.exists(_trk):
        shutil.copy(_trk, f"{WORK}/bytetrack_lg.yaml")
else:
    shutil.copytree(f"{DS}/code/litterguard", f"{WORK}/litterguard", dirs_exist_ok=True)
# prefer the config shipped WITH the kernel (immune to stale dataset snapshots)
_cfg_src = os.path.join(_here, "config.yaml")
if not os.path.exists(_cfg_src):
    _cfg_src = f"{DS}/code/config.yaml"
shutil.copy(_cfg_src, f"{WORK}/config.yaml")
sys.path.insert(0, WORK)

from litterguard.pipeline import Pipeline, load_config  # noqa: E402


def find_labels(video_path: str):
    """MIVIA label JSON: labels/ beside the videos/ folder (or beside video)."""
    d = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    for cand in [
        os.path.join(d, "labels", base + ".json"),
        os.path.join(os.path.dirname(d), "labels", base + ".json"),
        os.path.join(d, base + ".json"),
    ]:
        if os.path.exists(cand):
            return cand
    return None


def read_label(video_path: str):
    p = find_labels(video_path)
    if not p:
        return None
    try:
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        # MIVIA schema: {"label": "true"/"false" (string!), "timestamp": onset}
        if "label" in j:
            return str(j["label"]).strip().lower() == "true"
        if "Dumping" in j:
            return bool(j["Dumping"])
        return None
    except Exception:
        return None


def eval_videos(paths, shard_name):
    rows = []
    for v in paths:
        name = os.path.basename(v)
        label = read_label(v)
        row = {"video": name, "shard": shard_name, "label": label,
               "events": 0, "alerts": 0, "types": [], "guards": 0, "fps": 0.0, "error": None}
        try:
            cfg = load_config(os.path.join(WORK, "config.yaml"))
            cfg["runtime"]["device"] = "auto"
            cfg["models"]["gd_interval_s"] = 2.0
            cfg["models"]["gd_enable"] = True      # GPU: use the anchor     # GPU: anchor can keep up
            cfg["evidence"]["out_dir"] = os.path.join(WORK, "runs")
            pipe = Pipeline(cfg, source_override=v)
            pipe.run()
            row["events"] = pipe._stats["events"]
            row["alerts"] = sum(1 for e in pipe.engine.events if e.extras.get("alertable"))
            row["guards"] = len(pipe.engine.suppressed)
            dt = (pipe._stats["t0"] and pipe._stats["proc"]) or 0
            row["types"] = [e.type for e in pipe.engine.events]
            rows.append(row)
            print(f"[{shard_name}] {name}: label={label} events={row['events']} "
                  f"guards={row['guards']}", flush=True)
        except Exception as e:
            row["error"] = f"{e.__class__.__name__}: {e}"
            rows.append(row)
            print(f"[{shard_name}] {name}: ERROR {row['error']}", flush=True)
    return rows


def main():
    videos = sorted(
        glob.glob(f"{DS}/mivia_test/**/*.mp4", recursive=True) +
        glob.glob(f"{DS}/mivia_test/**/*.avi", recursive=True) +
        glob.glob(f"{DS}/realworld/*.mp4"))
    print(f"total videos: {len(videos)}", flush=True)

    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    print(f"GPUs visible: {n_gpu}", flush=True)

    if n_gpu >= 2:
        def worker(i, paths, q):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            q.put(eval_videos(paths, f"gpu{i}"))
        q = multiprocessing.Queue()
        half = (len(videos) + 1) // 2
        procs = [multiprocessing.Process(target=worker, args=(i, videos[:half] if i == 0 else videos[half:], q))
                 for i in range(2)]
        [p.start() for p in procs]
        rows = []
        for _ in procs:
            rows.extend(q.get())
        [p.join() for p in procs]
    else:
        rows = eval_videos(videos, "single")

    # ---- aggregate ----------------------------------------------------------
    with open(f"{WORK}/results.json", "w") as f:
        json.dump(rows, f, indent=1)
    tp = fp = fn = tn = unlabeled_events = 0
    for r in rows:
        if r["label"] is None:
            unlabeled_events += r["events"]
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
    atp = sum(1 for r in rows if r.get("label") and (r.get("alerts") or 0) > 0)
    afp = sum(1 for r in rows if r.get("label") is False and (r.get("alerts") or 0) > 0)
    summary = {"total": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "alert_tp": atp, "alert_fp": afp,
               "precision": round(tp / max(tp + fp, 1), 4),
               "alert_precision": round(atp / max(atp + afp, 1), 4),
               "recall": round(tp / max(tp + fn, 1), 4),
               "unlabeled_videos_events": unlabeled_events}
    with open(f"{WORK}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

"""Validate the departure gate on the three known ground-truth cases."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = r"C:\Users\krish\.zcode\workspace\697a15d4-0293-42c9-9dc6-78d11719a678\litter-guard"
PY = sys.executable
CASES = [
    ("litter1", os.path.join(ROOT, "media", "litter1.mp4"),
     "GT: real drop at ~32s (man sets black bag down, walks away) + carried bag at ~54s"),
    ("vid0085", os.path.join(ROOT, "media", "mivia_test", "videos", "vid0085.mp4"),
     "GT: true, dumping at t=16s"),
    ("vid0167", os.path.join(ROOT, "media", "mivia_test", "videos", "vid0167.mp4"),
     "GT: false (no dumping)"),
]


def run(case):
    name, path, note = case
    out = os.path.join(ROOT, "bench", "validate", name)
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "run.py", "--config", "config.yaml", "--source", path,
           "--out-dir", out, "--threads", "2", "--no-video"]
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=2400)
    evs = []
    jl = os.path.join(out, "events.jsonl")
    subs = [os.path.join(out, d, "events.jsonl") for d in sorted(os.listdir(out))
            if os.path.isdir(os.path.join(out, d))]
    for p in [jl] + subs:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line:
                    try:
                        evs.append(json.loads(line))
                    except Exception:
                        pass
    sup = []
    for d in sorted(os.listdir(out)):
        sp = os.path.join(out, d, "suppressed.jsonl")
        if os.path.exists(sp):
            for line in open(sp, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line:
                    try:
                        sup.append(json.loads(line))
                    except Exception:
                        pass
    return {"name": name, "note": note, "secs": round(time.time() - t0, 1),
            "events": [{"t": e.get("t"), "type": e.get("type"),
                        "alertable": (e.get("extras", {}) or {}).get("audit", {}).get("alertable"),
                        "depart_m": (e.get("extras", {}) or {}).get("audit", {}).get("depart_max_m"),
                        "culprit_m": (e.get("culprit") or {}).get("distance_m"),
                        "votes": e.get("votes")} for e in evs],
            "suppressed": sup}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r in ex.map(run, CASES):
            print(f"\n=== {r['name']} ({r['secs']}s) :: {r['note']}", flush=True)
            print("  events:", json.dumps(r["events"], indent=2), flush=True)
            reasons = {}
            for s in r["suppressed"]:
                reasons[s.get("reason")] = reasons.get(s.get("reason"), 0) + 1
            print("  suppressed reasons:", reasons, flush=True)

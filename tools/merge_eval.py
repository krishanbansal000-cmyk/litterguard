#!/usr/bin/env python
"""Merge seed + shard results, score overall and by event_type."""
import glob
import json
import os

rows = []
import glob as _g
for p in (["runs/mivia_local/results.json", "runs/mivia_final/results.json"] +
          sorted(_g.glob("runs/mivia_final/results_f*.json")) +
          sorted(_g.glob("runs/mivia_local/results_s*.json"))):
    if os.path.exists(p):
        rows.extend(json.load(open(p)))
seen, uniq = set(), []
for r in rows:
    if r["video"] not in seen:
        seen.add(r["video"])
        uniq.append(r)
rows = uniq

def score(rs):
    tp = fp = fn = tn = 0
    for r in rs:
        if r["label"] is None:
            continue
        pred = r["events"] > 0
        if r["label"] and pred: tp += 1
        elif r["label"] and not pred: fn += 1
        elif not r["label"] and pred: fp += 1
        else: tn += 1
    return tp, fp, fn, tn

tp, fp, fn, tn = score(rows)
overall = {"evaluated": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": round(tp / max(tp + fp, 1), 4),
           "recall": round(tp / max(tp + fn, 1), 4),
           "accuracy": round((tp + tn) / max(len([r for r in rows if r['label'] is not None]), 1), 4)}
by_type = {}
for r in rows:
    et = (r.get("meta") or {}).get("event_type") or "?"
    by_type.setdefault(et, []).append(r)
breakdown = {}
for et, rs in sorted(by_type.items()):
    tp, fp, fn, tn = score(rs)
    breakdown[et] = {"n": len(rs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "recall": round(tp / max(tp + fn, 1), 4)}
out = {"overall": overall, "by_event_type": breakdown,
       "missed_positives": [r["video"] for r in rows if r["label"] and r["events"] == 0],
       "false_positives": [r["video"] for r in rows if r["label"] is False and r["events"] > 0]}
print(json.dumps(out, indent=1))
json.dump({"rows": rows, "summary": out}, open("runs/mivia_local/final.json", "w"), indent=1)

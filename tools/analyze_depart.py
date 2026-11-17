#!/usr/bin/env python
"""Analyze recorded audit trails: depart_max for alert-grade events (TP vs FP)."""
import glob
import json
import os
import re

labels = {}
for p in glob.glob("runs/mivia_local/results_p*.json"):
    for r in json.load(open(p)):
        labels[r["video"]] = r["label"]

seen = set()
rows = []
for p in glob.glob("runs/fpcheck/*/events.jsonl") + glob.glob("runs/mivia_local/*/events.jsonl"):
    d = os.path.dirname(p)
    m = re.search(r"(vid\d+)", d)
    vid = m.group(1) + ".mp4" if m else None
    lab = labels.get(vid)
    for line in open(p, encoding="utf-8"):
        try:
            e = json.loads(line)
        except Exception:
            continue
        key = (d, e.get("event"))
        if key in seen:
            continue
        seen.add(key)
        a = e.get("extras", {}).get("audit", {})
        if a.get("depart_max_m") is not None:
            rows.append({"vid": vid, "label": lab, "type": e.get("type"),
                         "alerts": e.get("extras", {}).get("alertable"),
                         "depart": a["depart_max_m"],
                         "near_now": a.get("near_now_m")})

print(f"events with depart audit: {len(rows)}")
for r in sorted(rows, key=lambda x: (x["label"] is not True, x["depart"] or 999)):
    lab = "POS" if r["label"] else ("NEG" if r["label"] is False else "?")
    grade = "ALERT" if r["alerts"] else "review"
    print(f"  {lab:4} {grade:6} depart={r['depart']} near_now={r['near_now']} "
          f"{r['type']:12} {r['vid']}")

#!/usr/bin/env python
"""External review via OpenCode GO endpoint (subscription).

 - deepseek-v4.1-flash : reasoning review of LitterGuard's decisions (text)
 Requires: Authorization bearer, custom User-Agent, stable x-opencode-session.
"""
import base64
import json
import os
import sys
import urllib.request

BASE = "https://opencode.ai/zen/go/v1/chat/completions"
KEY = "oc_sk_662eaa1edb43_3VSgzBF0uB649nmB7GEHlVG7wR6KA13D"
SESSION_FILE = ".oc_session_uuid"


def session_uuid() -> str:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            s = f.read().strip()
        if s:
            return s
    import uuid
    s = str(uuid.uuid4())
    with open(SESSION_FILE, "w") as f:
        f.write(s)
    return s


def chat(model, messages, max_tokens=2000):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(BASE, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "User-Agent": "litterguard/0.1",
        "x-opencode-session": session_uuid()})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content", "")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "review"

    if mode == "review":
        # collect every event with its full internal audit trail
        events = []
        for root, _dirs, files in os.walk("runs"):
            if "events.jsonl" not in files:
                continue
            video = os.path.basename(root).split("_", 1)[-1]
            with open(os.path.join(root, "events.jsonl"), encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line)
                    e["video"] = video
                    events.append(e)
        compact = []
        for e in events:
            a = e.get("extras", {}).get("audit", {})
            compact.append({"video": e["video"], "type": e["type"], "t": e["t"],
                            "votes": e["votes"],
                            "culprit_dist_m": e["culprit"] and e["culprit"]["distance_m"],
                            "audit": a})
        prompt = f"""You are auditing a zero-training, CPU-only CCTV littering-detection system
(pre-trained models only: YOLO11n+ByteTrack persons, 2 litter detectors, MOG2 static-blob
physical proof, optional GroundingDINO). Its engine fires an event ONLY if:
litter track is NEW + STATIONARY (smoothed centroid sigma<15px, age>=5s) AND >=2 independent
votes (detector A / detector B / grounding-dino / large-object>=60kpx / held-release-chain)
AND a physical static blob exists at that spot AND a person track was within 3m of the SPAWN
point in [spawn-4s, fire+1s]. Exclusions: bin zones, pickup-suppression, 12s bootstrap,
90s cooldown. End-of-stream flush uses relaxed age>=2s.

FULL 12-VIDEO SUITE RESULTS, ROUND 2 (your prior audit was applied):
Ground truth: litter1-family = ONE real littering (person sets a tied bag on the ground
at t=32s; visually verified from evidence crops). Litter2-family = no littering (bag is
carried by a pedestrian the whole time; visually verified).
Round-1 audit found: a false positive family from an oversized near-full-frame detection
(a man STANDING at a bus stop HOLDING a bag; box contained the man himself ->
self-attribution at 0.12m). Fixes added, no retraining: (1) oversized-box guard
(box >35% of frame = rejected+logged), (2) person-overlap guard (litter box IoU>0.45
with a person = held item, rejected+logged), (3) audit trail now records age_rule
('fast_chain'=3s for held->release->landed chains, else 5s) and cooldown is
per-location (2m radius, 90s) NOT global.

ROUND-2 EVENTS (6 fired, all the same verified true positive across transforms):
{json.dumps(compact, indent=1)}

GUARD SUPPRESSIONS across suite (16 total, all logged with reason):
oversized_box: litter1@43s, scaled@14s+45s, reverse@47s+53s, dark@38s, fast@24s,
small@42s  -> these include the entire round-1 false-positive family
person_overlap: litter1@21s(carry phase), Litter2@21s+24s(carried bag), blur@22s,
small@22s  -> held/carried-bag detections suppressed
Negatives: ALL 7 no-littering runs fired 0 events. Reverse playback: 0 events.

CLOSING QUESTIONS (<=400 words, be decisive):
1) Are the round-1 findings now adequately resolved by the two guards + audit clarity?
2) Remaining top risk for the FIRST deployment false positive, and the cheapest guard
   against it that still needs NO training.
3) Final verdict for human-reviewed enforcement alerts: GO or NO-GO, with 1 condition."""
        print(chat("deepseek-v4.1-flash", [{"role": "user", "content": prompt}], 2500))


if __name__ == "__main__":
    main()

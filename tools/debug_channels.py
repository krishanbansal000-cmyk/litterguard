#!/usr/bin/env python
"""Stage-by-stage debug for one video: prints per-frame channel activity and
per-track kill reasons, so we can see exactly which detection step fails."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from litterguard.channels import (BackgroundChannel, LitterChannel,
                                  PersonVehicleChannel, center)
from litterguard.engine import LitterEngine
from litterguard.pipeline import load_config, open_source, resize_keep


def main(src):
    cfg = load_config("config.yaml")
    cap = open_source(src)
    persons_ch = PersonVehicleChannel(cfg["models"]["person"], cfg)
    litter_ch = LitterChannel(cfg["models"]["litter_a"], cfg["models"]["litter_b"], cfg)
    bg = BackgroundChannel(cfg)
    eng = LitterEngine(cfg, on_event=None)

    ok, f = cap.read()
    while not ok and f is None:
        ok, f = cap.read()
    last_pos = 0.0
    per = {"pa": 0, "la": 0, "lb": 0, "frames": 0}
    print(f"{'t':>5} {'persons':>7} {'litA':>5} {'litB':>5} {'bgBlobFrame':>10}")
    while True:
        ok, f = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos <= 0:
            continue
        if pos - last_pos < 0.2:
            continue
        last_pos = pos
        f = resize_keep(f, int(cfg["source"]["resize_max"]))
        persons, vehicles, personal = persons_ch.update(f, pos)
        dets = litter_ch.update(f)
        bg.update(f)
        na = sum(1 for d in dets if d.source == "litter_a")
        nb = sum(1 for d in dets if d.source == "litter_b")
        per["frames"] += 1
        per["pa"] = max(per["pa"], len(persons))
        per["la"] += na
        per["lb"] += nb
        print(f"{pos:>5.1f} {len(persons):>7} {na:>5} {nb:>5} {bg.frame_count:>10}")
        eng.update(pos, persons, vehicles, dets, bg, None, persons_ch.px_per_m,
                   personal=personal)

    print(f"\nTOTALS: frames={per['frames']} max_persons={per['pa']} "
          f"litter_a_dets={per['la']} litter_b_dets={per['lb']} "
          f"px_per_m={persons_ch.px_per_m:.1f}")
    print("\nlitter tracks:")
    for tid, tr in sorted(eng.litter.items()):
        if len(tr.boxes) < 2:
            continue
        box = np.median(np.array(list(tr.boxes)[-12:]), axis=0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        reasons = []
        if tr.age < 5.0:
            reasons.append("age<5")
        if not tr.stationary:
            reasons.append("moving")
        if len(tr.sources) < 2:
            reasons.append(f"votes={sorted(tr.sources)}")
        if area < 250:
            reasons.append("tiny")
        if tr.pre_existing:
            reasons.append("pre_existing")
        near = eng._nearest_person(tr, tr.last_t + 0.001, back=30, fwd=0)
        bs = list(tr.boxes)[-12:]
        ious = [1.0] * 0
        from litterguard.channels import xyxy_iou as _xiou
        ious = [_xiou(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]
        miou = sum(ious) / len(ious) if ious else 0.0
        print(f"  #{tid} t=[{tr.first_t:.1f}..{tr.last_t:.1f}] age={tr.age:.1f} "
              f"stat={tr.stationary} miou={miou:.2f} area={area:.0f} src={sorted(tr.sources)} "
              f"chain={tr.chain_armed}/{tr.chain_source} picked={tr.picked} "
              f"near={near and (near[0], round(near[1], 1))} "
              f"BLOCK={reasons or 'PASSED'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "media/mivia_test/videos/vid0044.mp4")

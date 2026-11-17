#!/usr/bin/env python
"""Debug: replay a video through channels+engine and dump per-track state so we
can see exactly which rule blocks events."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from litterguard.channels import (BackgroundChannel, GroundingDINOChannel,
                                  LitterChannel, PersonVehicleChannel, center)
from litterguard.engine import LitterEngine
from litterguard.pipeline import load_config, open_source, resize_keep


def main(src):
    cfg = load_config("config.yaml")
    cap = open_source(src)
    persons_ch = PersonVehicleChannel(cfg["models"]["person"], cfg)
    litter_ch = LitterChannel(cfg["models"]["litter_a"], cfg["models"]["litter_b"], cfg)
    bg = BackgroundChannel(cfg)
    gd = GroundingDINOChannel(cfg["models"]["grounding_dino"], cfg)
    gd.start()
    eng = LitterEngine(cfg, on_event=None)

    last_proc = -1e9
    last_pos = 0.0
    per_frame_persons = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos <= 0:
            continue
        last_pos = max(last_pos, pos)
        if pos - last_proc < 0.2:
            continue
        last_proc = pos
        f = resize_keep(f, int(cfg["source"]["resize_max"]))
        persons, vehicles = persons_ch.update(f, pos)
        per_frame_persons = max(per_frame_persons, len(persons))
        dets = litter_ch.update(f)
        bg.update(f)
        gd.submit(f, pos)
        eng.update(pos, persons, vehicles, dets, bg, gd, persons_ch.px_per_m)

    fired = eng.flush(last_pos, bg=bg, gd=gd)
    for item in fired:
        ev = item["event"]
        print(f"[FLUSH-EVENT] {ev.type} t={ev.t:.1f} votes={ev.votes} "
              f"culprit={ev.culprit and ev.culprit['track_id']} box={[round(v) for v in ev.litter_box]}")

    print(f"\nGD available={gd.available} votes={len(gd.votes)}")
    print(f"px_per_m={persons_ch.px_per_m:.1f} max_persons_frame={per_frame_persons}")
    print(f"{'id':>3} {'first':>6} {'last':>6} {'age':>5} {'stat':>5} {'area':>7} "
          f"{'votes':>22} {'pre':>4} {'fired':>5} {'pick':>4} blob near")
    for tid, tr in sorted(eng.litter.items()):
        if len(tr.boxes) < 3:
            continue
        box = np.median(np.array(list(tr.boxes)[-15:]), axis=0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        votes = set(tr.sources)
        gv = gd.vote_near(tuple(box), tr.last_t, ttl=45)
        if gv:
            votes.add("gd")
        blob = bg.blob_at(tuple(box)) if len(tr.boxes) >= 8 else None
        near = eng._nearest_person(tr, tr.last_t + 0.001, back=30, fwd=0)
        print(f"{tid:>3} {tr.first_t:>6.1f} {tr.last_t:>6.1f} {tr.age:>5.1f} "
              f"{str(tr.stationary):>5} {area:>7.0f} {str(sorted(votes)):>22} "
              f"{str(tr.pre_existing):>4} {str(tr.fired):>5} {str(tr.picked):>4} "
              f"{blob} near={near and (near[0], round(near[1],1))}")
        if tr.age > 1.5 and len(tr.centers) >= 6:
            cs = [f"({c[0]:.0f},{c[1]:.0f})" for c in list(tr.centers)[-12:]]
            print(f"      centers: {' '.join(cs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "media/litter1.mp4")

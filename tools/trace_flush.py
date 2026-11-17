#!/usr/bin/env python
"""Replay one video, then evaluate EVERY gate manually for each surviving litter
track at end-of-stream, printing pass/fail per gate. Pinpoints the blocker."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from litterguard.channels import BackgroundChannel, GroundingDINOChannel, \
    LitterChannel, PersonVehicleChannel, xyxy_iou  # noqa: E402
from litterguard.engine import LitterEngine, _smoothed_sigma, _point_in_box  # noqa: E402
from litterguard.pipeline import load_config, open_source, resize_keep  # noqa: E402


def main(src):
    cfg = load_config("config.yaml")
    cap = open_source(src)
    persons_ch = PersonVehicleChannel(cfg["models"]["person"], cfg)
    litter_ch = LitterChannel(cfg["models"]["litter_a"], cfg["models"]["litter_b"], cfg)
    bg = BackgroundChannel(cfg)
    eng = LitterEngine(cfg, on_event=None)
    ok, f = cap.read()
    last = 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos <= 0 or pos - last < 0.2:
            continue
        last = pos
        f = resize_keep(f, int(cfg["source"]["resize_max"]))
        persons, vehicles = persons_ch.update(f, pos)
        dets = litter_ch.update(f)
        bg.update(f)
        eng.update(pos, persons, vehicles, dets, bg, None, persons_ch.px_per_m)
    t = last
    logic = cfg["logic"]
    frame_area = 960 * 540
    print(f"\n=== gate-by-gate at end-of-stream t={t:.1f} ===")
    for tid, tr in sorted(eng.litter.items()):
        if tr.age < 3 or len(tr.boxes) < 8 or tr.fired:
            continue
        box = tuple(np.median(np.array(list(tr.boxes)[-15:]), axis=0).tolist())
        area = (box[2] - box[0]) * (box[3] - box[1])
        bs = list(tr.boxes)[-12:]
        ious = [xyxy_iou(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]
        miou = sum(ious) / len(ious) if ious else 0.0
        votes = set(tr.sources)
        r = bg.blob_ratio(box)
        near = eng._nearest_person(tr, tr.first_t, back=4.0, fwd=(t - tr.first_t) + 1.0)
        near_now = None
        for _pid, phist in eng.person_hist.items():
            for (ht, c, _b) in phist:
                if ht >= t - 1.5:
                    d = eng._dist_m(c, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                    if near_now is None or d < near_now:
                        near_now = d
        pmax = float(logic.get("person_containment_max_iou", 0.45))
        overlap = any(xyxy_iou(box, b) > pmax for (_p, _d, b) in eng._persons_near(tr, t))
        print(f"\ntrack #{tid} t=[{tr.first_t:.1f}..{tr.last_t:.1f}] age={tr.age:.1f} "
              f"area={area:.0f} src={sorted(votes)} pre={tr.pre_existing} picked={tr.picked}")
        print(f"  age>=2(flush)      : {'PASS' if tr.age >= 2 else 'FAIL'}")
        print(f"  picked             : {'PASS' if not tr.picked else 'FAIL (picked up)'}")
        print(f"  pre_existing       : {'PASS' if not tr.pre_existing else 'FAIL'}")
        print(f"  stationary         : {'PASS' if (miou >= 0.35 or _smoothed_sigma(list(tr.centers)) < 20) else f'FAIL (miou={miou:.2f})'}")
        print(f"  area>=min          : {'PASS' if area >= 250 else 'FAIL'}")
        if area > 0.35 * frame_area:
            inside = any(_point_in_box(((b[0]+b[2])/2, (b[1]+b[3])/2), box)
                         for (_p, _d, b) in eng._persons_near(tr, t))
            print(f"  oversized          : {'PASS' if not inside else f'FAIL (person inside={inside})'}")
        print(f"  person_overlap     : {'PASS' if not overlap else 'FAIL'}")
        print(f"  votes>=2           : {'PASS' if len(votes) >= 2 else f'FAIL {sorted(votes)}'}")
        print(f"  blob ratio={r:.2f} : {'PASS' if (r >= 0.25 or r*area >= 8000) else 'FAIL'}")
        h, w = bg.static_mask.shape[:2]
        x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
        x2, y2 = int(min(w, box[2])), int(min(h, box[3]))
        reg = bg.static_mask[y1:y2, x1:x2]
        print(f"    [blob debug] frames={bg.frame_count} min={bg.min_frames} "
              f"mask_shape={bg.static_mask.shape} box=({x1},{y1},{x2},{y2}) "
              f"region_mean={reg.mean():.3f} region_max={reg.max():.2f} "
              f"frac>0.55={(reg > 0.55).mean():.3f}")
        print(f"  culprit near       : {near and (near[0], round(near[1],2))}")
        print(f"  near_now (departed): {near_now and round(near_now,2)}")


if __name__ == "__main__":
    main(sys.argv[1])

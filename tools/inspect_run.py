#!/usr/bin/env python
"""Run one video through the real pipeline and dump every litter track's final
state + all suppressions + events, so the exact blocking rule is visible."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litterguard.pipeline import Pipeline, load_config  # noqa: E402
from litterguard.engine import _smoothed_sigma  # noqa: E402
from litterguard.channels import xyxy_iou  # noqa: E402
import numpy as np  # noqa: E402


def main(path):
    cfg = load_config("config.yaml")
    cfg["evidence"]["out_dir"] = "runs/inspect"
    pipe = Pipeline(cfg, source_override=path)
    pipe.run()
    eng = pipe.engine
    print("\n=== suppressions ===")
    for s in eng.suppressed:
        print(" ", s)
    print("=== events ===")
    for e in eng.events:
        print(f"  t={e.t:.1f} type={e.type} votes={e.votes} culprit={e.culprit}")
    print("=== litter tracks (age>2) ===")
    for tid, tr in sorted(eng.litter.items()):
        if tr.age <= 2 or len(tr.boxes) < 4:
            continue
        box = np.median(np.array(list(tr.boxes)[-12:]), axis=0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        bs = list(tr.boxes)[-12:]
        ious = [xyxy_iou(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]
        miou = sum(ious) / len(ious) if ious else 0.0
        near = eng._nearest_person(tr, tr.last_t + 0.001, back=30, fwd=0)
        # where is the nearest person NOW (last 1.5s)?
        t_last = tr.last_t
        near_now = None
        for _pid, phist in eng.person_hist.items():
            for (ht, c, _b) in phist:
                if ht >= t_last - 1.5:
                    d = eng._dist_m(c, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                    if near_now is None or d < near_now:
                        near_now = d
        print(f"  #{tid} t=[{tr.first_t:.1f}..{tr.last_t:.1f}] age={tr.age:.1f} "
              f"area={area:.0f} miou={miou:.2f} sigma={_smoothed_sigma(list(tr.centers)):.0f} "
              f"stat={tr.stationary} src={sorted(tr.sources)} fired={tr.fired} "
              f"picked={tr.picked} pre={tr.pre_existing} chain={tr.chain_armed} "
              f"near_hist={near and (near[0], round(near[1], 2))} near_now={near_now and round(near_now, 2)}")


if __name__ == "__main__":
    main(sys.argv[1])

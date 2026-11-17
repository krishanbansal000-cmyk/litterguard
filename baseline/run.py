#!/usr/bin/env python
"""LitterGuard CLI.

  python run.py --config config.yaml [--source media/litter1.mp4] [--show] [--record]
"""
import argparse
import sys

from litterguard.pipeline import Pipeline, load_config


def main():
    ap = argparse.ArgumentParser(description="Zero-training local littering detection")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None, help="video file | webcam index | rtsp://")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--record", action="store_true", help="save annotated video of the run")
    ap.add_argument("--out-dir", default=None, help="override evidence output dir")
    ap.add_argument("--threads", type=int, default=None, help="torch thread count")
    ap.add_argument("--proc-fps", type=float, default=None, help="stage-1 processing fps")
    ap.add_argument("--no-video", action="store_true", help="skip annotated video dump")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["source"]["input"] = args.source
    if args.out_dir:
        cfg["evidence"]["out_dir"] = args.out_dir
    if args.threads:
        cfg["runtime"]["torch_threads"] = args.threads
    if args.proc_fps:
        cfg["source"]["proc_fps"] = args.proc_fps
    if args.no_video:
        cfg["evidence"]["save_annotated_video"] = False

    try:
        pipe = Pipeline(cfg, show=args.show, record=args.record)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    pipe.run()


if __name__ == "__main__":
    main()

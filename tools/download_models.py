#!/usr/bin/env python
"""Parallel download of all pre-trained models (nothing here is trained)."""
from __future__ import annotations

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

URLS = {
    # official Ultralytics release assets
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "yolo11n-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    "yolov8s-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
}
HF_MODELS = {
    # community litter YOLOv8 (TACO-style, trained by others, MIT repo)
    "esapzoi/litter-detection-yolov8": ["best.pt", "model.pt", "weights/best.pt"],
}
GD_HF = "IDEA-Research/grounding-dino-base"


def _fetch(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000_000:
        print(f"[skip] {os.path.basename(dest)} already present")
        return dest
    print(f"[get ] {os.path.basename(dest)} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    print(f"[done] {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.0f} MB)")
    return dest


def _hf_try(repo: str, names, dest_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    for fn in names:
        try:
            p = hf_hub_download(repo_id=repo, filename=fn, local_dir=dest_dir)
            print(f"[done] {repo}/{fn}")
            return p
        except Exception:
            continue
    raise RuntimeError(f"no weight file found in {repo} (tried {names})")


def _warm_grounding_dino():
    """GroundingDINO downloads into the HF cache via transformers."""
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    AutoProcessor.from_pretrained(GD_HF)
    GroundingDinoForObjectDetection.from_pretrained(GD_HF)
    print("[done] grounding-dino-base cached (HF cache)")


def main():
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, url in URLS.items():
            jobs.append(ex.submit(_fetch, url, os.path.join(MODELS, name)))
        for repo, names in HF_MODELS.items():
            jobs.append(ex.submit(_hf_try, repo, names, os.path.join(MODELS, "litter-detection-yolov8")))
        jobs.append(ex.submit(_warm_grounding_dino))
        for j in jobs:
            try:
                j.result()
            except Exception as e:
                print(f"[warn] {e.__class__.__name__}: {e}", file=sys.stderr)
    print("\nAll model downloads attempted. See messages above for status.")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Convert TACO (COCO-style) to YOLO detection format and launch fine-tune."""
import json
import os
import shutil
import sys
from collections import Counter

SRC = "media/taco_images/data"
OUT = "datasets/taco_yolo"
EPOCHS = int(os.environ.get("EPOCHS", "20"))
IMGSZ = 640

ann = json.load(open(os.path.join(SRC, "annotations.json"), encoding="utf-8"))
cats = {c["id"]: c["name"] for c in ann["categories"]}
print("TACO categories:", Counter(c["name"] for c in ann["categories"]))
print("images:", len(ann["images"]), "annotations:", len(ann["annotations"]))

# single-class 'litter' fine-tune: any TACO instance = litter
for split in ("train", "val"):
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)

img_by_id = {im["id"]: im for im in ann["images"]}
anns_by_img = {}
for a in ann["annotations"]:
    anns_by_img.setdefault(a["image_id"], []).append(a)

n_train, n_val = 0, 0
img_list = list(ann["images"])
for k, im in enumerate(img_list):
    fn = im["file_name"]
    src = os.path.join(SRC, fn)
    if not os.path.exists(src):
        continue
    split = "val" if (k % 10 == 0) else "train"
    w, h = im["width"], im["height"]
    dst_img = f"{OUT}/{split}/images/{os.path.basename(fn)}"
    shutil.copy(src, dst_img)
    lines = []
    for a in anns_by_img.get(im["id"], []):
        bx, by, bw, bh = a["bbox"]
        xc, yc = (bx + bw / 2) / w, (by + bh / 2) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")
    with open(f"{OUT}/{split}/labels/{os.path.splitext(os.path.basename(fn))[0]}.txt", "w") as f:
        f.write("\n".join(lines))
    if split == "train":
        n_train += 1
    else:
        n_val += 1

with open(f"{OUT}/taco_litter.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: train/images\nval: val/images\n"
            f"names:\n  0: litter\n")
print(f"converted: train={n_train} val={n_val}")
print("yaml:", f"{OUT}/taco_litter.yaml")

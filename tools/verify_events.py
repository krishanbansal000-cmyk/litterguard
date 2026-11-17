"""Extract frames around detected events to verify true positives.
Builds a labeled contact sheet from the raw clip and the annotated run video.
"""
import os, sys
import cv2
import numpy as np

ROOT = r"C:\Users\krish\.zcode\workspace\697a15d4-0293-42c9-9dc6-78d11719a678\litter-guard"
RUN = os.path.join(ROOT, "runs", "20261115-154712_litter1")
RAW = os.path.join(ROOT, "media", "litter1.mp4")
ANN = os.path.join(RUN, "annotated_full.mp4")
OUT = os.path.join(ROOT, "media", "verify_events.jpg")

EVENTS = [("ev001 chain_drop", 32.1, [30.0, 31.5, 32.1, 33.5, 36.0]),
          ("ev002 drop", 54.1, [52.0, 53.5, 54.1, 55.5, 58.0])]


def grab(path, times):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = {}
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if ok:
            out[round(t, 1)] = fr
    cap.release()
    return out


def sheet(frames, labels, path, width=430):
    tiles = []
    for (t, fr), lab in zip(frames, labels):
        if fr is None:
            continue
        h = int(fr.shape[0] * width / fr.shape[1])
        fr = cv2.resize(fr, (width, h))
        cv2.rectangle(fr, (0, 0), (width, 26), (0, 0, 0), -1)
        cv2.putText(fr, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(fr)
    if not tiles:
        print("no frames")
        return
    maxh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, maxh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    grid = np.hstack(tiles)
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", path, grid.shape)


if __name__ == "__main__":
    times = [t for _, _, ts in EVENTS for t in ts]
    labels = [f"{name} t={t:.1f}s" for name, _, ts in EVENTS for t in ts]
    raw = grab(RAW, times)
    sheet([(t, raw.get(round(t, 1))) for t in times], labels,
          os.path.join(ROOT, "media", "verify_raw.jpg"))
    if os.path.exists(ANN):
        ann = grab(ANN, times)
        sheet([(t, ann.get(round(t, 1))) for t in times], labels,
              os.path.join(ROOT, "media", "verify_annotated.jpg"))
    else:
        print("no annotated video")

"""Zoomed crops around detected events to judge true/false positive by eye."""
import os
import cv2
import numpy as np

ROOT = r"C:\Users\krish\.zcode\workspace\697a15d4-0293-42c9-9dc6-78d11719a678\litter-guard"
RAW = os.path.join(ROOT, "media", "litter1.mp4")
OUT = os.path.join(ROOT, "media", "verify_zoom.jpg")

# (label, time, crop rect x1,y1,x2,y2 in 960x540 space)
SHOTS = [
    ("ev001 t=30.0", 30.0, (120, 60, 620, 420)),
    ("ev001 t=32.1", 32.1, (120, 60, 620, 420)),
    ("ev001 t=34.0", 34.0, (120, 60, 620, 420)),
    ("ev002 t=52.0", 52.0, (60, 40, 560, 400)),
    ("ev002 t=54.1", 54.1, (60, 40, 560, 400)),
    ("ev002 t=56.5", 56.5, (60, 40, 560, 400)),
]

cap = cv2.VideoCapture(RAW)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
tiles = []
for label, t, (x1, y1, x2, y2) in SHOTS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    if not ok:
        continue
    h, w = fr.shape[:2]
    sx, sy = w / 960.0, h / 540.0
    crop = fr[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
    crop = cv2.resize(crop, (520, 360), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (520, 24), (0, 0, 0), -1)
    cv2.putText(crop, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(crop)
cap.release()

grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]) if len(tiles) == 6 else np.hstack(tiles)
cv2.imwrite(OUT, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", OUT, grid.shape)

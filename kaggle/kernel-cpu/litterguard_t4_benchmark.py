"""
LitterGuard benchmark — full-accuracy channels, runs on Kaggle (CPU or T4).

Robust input discovery:
  1. any *.mp4/*.avi mounted under /kaggle/input (Kaggle dataset attach)
  2. else download the base clip from the upstream repo
  3. stress variants (dark / blur / fast) are synthesized in-kernel if absent

Outputs: metrics.json + annotated frames -> kernel output.
"""
import glob, json, os, subprocess, time

print("== installing deps ==", flush=True)
subprocess.run(["pip", "install", "-q", "ultralytics", "supervision",
                "transformers", "accelerate"], check=False)

import torch, cv2, numpy as np
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

WORK = "/kaggle/working"
MEDIA = os.path.join(WORK, "media")
os.makedirs(MEDIA, exist_ok=True)

VOCAB = ["litter", "trash", "garbage bag", "plastic bottle", "bottle", "can",
         "cup", "cardboard box", "food wrapper", "plastic bag", "trash bag",
         "bin bag", "black bag", "styrofoam container", "discarded clothes",
         "trash on the ground"]
GD_LABELS = ["litter", "trash", "garbage bag", "plastic bag", "trash bag", "rubbish"]
BASE_URLS = [
    "https://github.com/LitterBugCam/LitterBug-Algorithm/raw/Desktop_Litterbug/litter1.mp4",
    "https://raw.githubusercontent.com/LitterBugCam/LitterBug-Algorithm/Desktop_Litterbug/litter1.mp4",
]

results = {"input_discovery": {}}

# ---------------- input discovery ----------------
print("\n== discovering input videos ==", flush=True)
found = sorted(glob.glob("/kaggle/input/**/*.mp4", recursive=True) +
               glob.glob("/kaggle/input/**/*.avi", recursive=True))
print("mounted files:", found, flush=True)
try:
    print("kaggle/input listing:", os.listdir("/kaggle/input"), flush=True)
except Exception as e:
    print("input listing failed:", repr(e), flush=True)

videos = {}
for f in found:
    key = os.path.basename(f).split(".")[0]
    videos[key] = f

base = None
if not videos:
    print("no mounted video -> downloading base clip", flush=True)
    base = os.path.join(MEDIA, "litter1.mp4")
    for url in BASE_URLS:
        try:
            subprocess.run(["wget", "-q", "-O", base, url], check=False, timeout=180)
            if os.path.exists(base) and os.path.getsize(base) > 100_000:
                print("downloaded from", url, os.path.getsize(base), flush=True)
                break
        except Exception as e:
            print("download failed:", repr(e), flush=True)
    if not (os.path.exists(base) and os.path.getsize(base) > 100_000):
        print("DOWNLOAD FAILED", flush=True)
        videos = {}
    else:
        videos = {"normal": base}
else:
    base = videos.get("normal") or next(iter(videos.values()))

# synthesize stress variants if missing
def synth(src, name, mode):
    dst = os.path.join(MEDIA, f"{name}.mp4")
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if mode == "dark":
            fr = (np.clip((fr.astype(np.float32) / 255.0) ** 2.4 * 190, 0, 255)).astype(np.uint8)
            fr = cv2.addWeighted(fr, 0.9, np.full_like(fr, 6), 0.1, 0)
        elif mode == "blur":
            fr = cv2.GaussianBlur(fr, (11, 11), 0)
        elif mode == "fast":
            if i % 2:
                i += 1; continue
        out.write(fr); i += 1
    out.release(); cap.release()
    return dst

if base and os.path.exists(base):
    for mode in ("dark", "blur", "fast"):
        key = f"normal_{mode}"
        if key not in videos:
            try:
                videos[key] = synth(base, f"litter1_{mode}", mode)
                print("synthesized", videos[key], flush=True)
            except Exception as e:
                print("synth failed", mode, repr(e), flush=True)

results["input_discovery"] = {"videos": {k: v for k, v in videos.items()},
                              "mounted_count": len(found)}
print("videos to test:", list(videos.keys()), flush=True)

# ---------------- fast channels ----------------
print("\n== fast channels (YOLO11n person + YOLO-World litter) ==", flush=True)
from ultralytics import YOLO
person_model = YOLO("yolo11n.pt")
world = YOLO("yolov8s-worldv2.pt")
world.set_classes(VOCAB)

def run_fast(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot open video"}
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps_in / 5)))          # ~5 FPS sampling
    frames_seen = n_world = n_person = 0
    t0 = time.time()
    ok, frame = cap.read()
    while ok:
        if frames_seen % step == 0:
            rp = person_model.predict(frame, verbose=False, classes=[0])[0]
            rw = world.predict(frame, verbose=False, conf=0.24)[0]
            n_person += len(rp.boxes)
            n_world += len(rw.boxes)
        frames_seen += 1
        ok, frame = cap.read()
    dt = time.time() - t0
    cap.release()
    processed = max(1, -(-frames_seen // step))
    return {"frames": frames_seen, "infer_frames": processed,
            "wall_s": round(dt, 1), "fps": round(processed / dt, 1),
            "person_boxes_total": n_person, "litter_boxes_total": n_world}

for name, path in videos.items():
    if not os.path.exists(path):
        continue
    print("fast:", name, flush=True)
    results[f"fast_{name}"] = run_fast(path)
    print(results[f"fast_{name}"], flush=True)

# ---------------- accuracy anchor: GroundingDINO ----------------
print("\n== GroundingDINO-base (transformers zero-shot pipeline) ==", flush=True)
try:
    from transformers import pipeline as hf_pipeline
    gd = hf_pipeline("zero-shot-object-detection",
                     model="IDEA-Research/grounding-dino-base",
                     device=0 if torch.cuda.is_available() else -1)
    from PIL import Image
    cap = cv2.VideoCapture(base)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps_in * 4)))          # every ~4 s (CPU-friendly)
    idx = n = det_total = agree = 0
    frame_times = []
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tt = time.time()
            dets = gd(img, candidate_labels=GD_LABELS, threshold=0.25, text_threshold=0.24)
            frame_times.append(time.time() - tt)
            det_total += len(dets)
            rw = world.predict(frame, verbose=False, conf=0.24)[0]
            wboxes = rw.boxes.xyxy.cpu().numpy() if len(rw.boxes) else []
            for d in dets:
                b = d["box"]
                box = np.array([b["xmin"], b["ymin"], b["xmax"], b["ymax"]])
                for wb in wboxes:
                    x1, y1 = max(box[0], wb[0]), max(box[1], wb[1])
                    x2, y2 = min(box[2], wb[2]), min(box[3], wb[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    union = ((box[2]-box[0])*(box[3]-box[1]) +
                             (wb[2]-wb[0])*(wb[3]-wb[1]) - inter)
                    if union > 0 and inter / union > 0.3:
                        agree += 1
                        break
            n += 1
            if n <= 3 and dets:
                vis = frame.copy()
                for d in dets:
                    b = d["box"]
                    cv2.rectangle(vis, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), (0, 0, 255), 2)
                    cv2.putText(vis, f'{d["label"]} {d["score"]:.2f}',
                                (b["xmin"], max(20, b["ymin"]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(WORK, f"gd_frame_{n}.jpg"), vis)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    results["grounding_dino"] = {
        "frames_sampled": n, "detections_total": det_total,
        "avg_s_per_frame": round(float(np.mean(frame_times)), 3) if frame_times else None,
        "fps": round(1.0/float(np.mean(frame_times)), 2) if frame_times else None,
        "agreement_with_yolo_world": agree}
    print(results["grounding_dino"], flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    results["grounding_dino"] = {"error": repr(e)}
    print("GD failed:", repr(e), flush=True)

with open(os.path.join(WORK, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE — metrics.json written", flush=True)
print(json.dumps(results, indent=2), flush=True)

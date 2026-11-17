#!/usr/bin/env bash
# Deploy LitterGuard eval to Kaggle: stage dataset, upload, push GPU kernel.
set -e
PY="/c/Users/krish/AppData/Local/Programs/Python/Python312/python.exe"
export PATH="$PATH:/c/Users/krish/AppData/Local/Programs/Python/Python312/Scripts"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/kaggle_pkg/dataset_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/code/litterguard" "$STAGE/weights" "$STAGE/mivia_test" "$STAGE/realworld"

# code + config (paths rewritten for Kaggle mount)
cp "$ROOT"/litterguard/*.py "$STAGE/code/litterguard/"
cp "$ROOT/models/bytetrack_lg.yaml" "$STAGE/code/"
sed -e 's#models/yolo11n.pt#/kaggle/input/litterguard-eval/weights/yolo11n.pt#' \
    -e 's#models/yolo11n-pose.pt#/kaggle/input/litterguard-eval/weights/yolo11n-pose.pt#' \
    -e 's#models/litter-detection-yolov8/best.pt#/kaggle/input/litterguard-eval/weights/litter_yolov8.pt#' \
    -e 's#models/yolov8s-worldv2.pt#/kaggle/input/litterguard-eval/weights/yolov8s-worldv2.pt#' \
    -e 's#models/extra/waste-detection-yolov8/best.pt#/kaggle/input/litterguard-eval/weights/extra_waste.pt#' \
    -e 's#models/extra/yolov8s-garbage-detection/best_yolov8s_final.pt#/kaggle/input/litterguard-eval/weights/extra_garbage.pt#' \
    -e 's#models/extra/littercam-yolov9c/best.pt#/kaggle/input/litterguard-eval/weights/extra_littercam.pt#' \
    -e 's#models/bytetrack_lg.yaml#/kaggle/input/litterguard-eval/code/bytetrack_lg.yaml#' \
    -e 's#person_tracker_yaml: "models/bytetrack_lg.yaml"#person_tracker_yaml: "/kaggle/input/litterguard-eval/code/bytetrack_lg.yaml"#' \
    "$ROOT/config.yaml" > "$STAGE/code/config.yaml"

# small weights (GD + pose pulled from HF inside the kernel; tracker yaml above)
cp "$ROOT/models/yolo11n.pt" "$STAGE/weights/"
cp "$ROOT/models/yolov8s-worldv2.pt" "$STAGE/weights/"
cp "$ROOT/models/litter-detection-yolov8/best.pt" "$STAGE/weights/litter_yolov8.pt"
cp "$ROOT/models/yolo11n-pose.pt" "$STAGE/weights/"
cp "$ROOT/models/extra/waste-detection-yolov8/best.pt" "$STAGE/weights/extra_waste.pt" 2>/dev/null || true
cp "$ROOT/models/extra/yolov8s-garbage-detection/best_yolov8s_final.pt" "$STAGE/weights/extra_garbage.pt" 2>/dev/null || true
cp "$ROOT/models/extra/littercam-yolov9c/best.pt" "$STAGE/weights/extra_littercam.pt" 2>/dev/null || true

# videos: MIVIA test set + real-world clips
if [ -d "$ROOT/media/mivia_test" ]; then cp -r "$ROOT/media/mivia_test/." "$STAGE/mivia_test/"; fi
cp "$ROOT"/media/mivia_new/*.mp4 "$STAGE/realworld/" 2>/dev/null || true

cp "$ROOT/kaggle_pkg/dataset-metadata.json" "$STAGE/"
echo "== staged:"; du -sm "$STAGE" | cut -f1; echo MB
find "$STAGE" -maxdepth 2 -type d | head

# upload dataset (create or version)
if kaggle datasets status krishanbansal/litterguard-eval >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" --dir-mode zip -m "update videos" >/dev/null
  echo "dataset versioned"
else
  kaggle datasets create -p "$STAGE" --dir-mode zip >/dev/null
  echo "dataset created"
fi

# push kernel
cp "$ROOT/kaggle_pkg/kernel-metadata.json" "$ROOT/kaggle_pkg/litterguard-gpu-eval.py" /tmp/ 2>/dev/null || true
kaggle kernels push -p "$ROOT/kaggle_pkg" && echo "kernel pushed"

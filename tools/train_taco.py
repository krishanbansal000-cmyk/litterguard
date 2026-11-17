#!/usr/bin/env python
"""Fine-tune yolo11n on TACO litter images (CPU, unattended)."""
import torch
from ultralytics import YOLO

torch.set_num_threads(14)
model = YOLO("models/yolo11n.pt")
model.train(
    data="datasets/taco_yolo/taco_litter.yaml",
    epochs=int(__import__("os").environ.get("EPOCHS", "25")),
    imgsz=640,
    batch=8,
    device="cpu",
    workers=6,
    project="runs/taco_finetune",
    name="taco_yolo11n",
    exist_ok=True,
    plots=True,
)
print("BEST WEIGHTS:", "runs/taco_finetune/taco_yolo11n/weights/best.pt")

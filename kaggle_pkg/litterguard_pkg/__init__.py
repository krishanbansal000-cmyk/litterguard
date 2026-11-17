"""LitterGuard — zero-training, fully-local littering event detection.

Channels (independent evidence sources):
  A) persons + vehicles  : YOLO11n COCO, tracked (ByteTrack)
  B) litter detector #1  : community TACO-style YOLOv8 (esapzoi)
  C) litter detector #2  : YOLO-World zero-shot custom vocabulary
  D) physical proof      : MOG2 static-foreground blobs (training-free)
  E) slow anchor         : GroundingDINO-base zero-shot, own thread

Engine: temporal logic (new + stationary + agreement + proximity + exclusions),
held->release->landed chain, vehicle-eject, pickup suppression.
"""

__version__ = "0.1.0"

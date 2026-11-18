# Problem Statements & Detector Ideas — Smart Cities, Prisons, Border Forces

Derived from the LitterGuard architecture (detector ensemble + tracking +
temporal logic + evidence packs + VLM verification). Each problem reuses
**>80% of the existing codebase** — the channels, engine, evidence, and
verification stages are domain-agnostic.

---

## 1. Smart City — Bin Overflow & Missed-Collection Detection

**Problem:** Municipal corporations don't know which bins are overflowing until
complaints arrive. Collection routes are static while fill-levels vary daily.

**Detector:** Camera at each bin site → fill-level classifier (segmentation on
the bin-opening region; pre-trained SAM works out of the box) → alert when
>80% full for >4 hours → route-optimization feed. This is the inverse of
LitterGuard's "unattended object" logic: the *absence* of collection is the event.

**Data:** None needed — the camera calibrates itself (empty bin = baseline,
full bin = threshold).

**New work:** ~20 lines (fill-level classifier).

**Civic value:** 20–30% fuel savings on optimized collection routes (proven
in Barcelona/Seoul pilots).

---

## 2. Smart City — Encroachment & Street-Obstruction Detection

**Problem:** Illegal hawkers, parked vehicles, construction material, and
dumped debris obstruct roads and footpaths. Complaints are reactive; municipal
action takes weeks.

**Detector:** Fixed camera on a known zone → object detection + persistence
(LitterGuard's track logic) → if a non-pedestrian object occupies the
carriageway/footway for >N minutes → alert with zone classification (road vs
footpath vs parking).

**Data:** COCO classes cover vehicles; the stationarity test (IoU-stability)
does the rest.

**New work:** Zone polygons only (config field, no code).

**Civic value:** BMC/BBMP get proactive encroachment maps instead of
complaint queues.

---

## 3. Prisons — Contraband Throw-Detection over Perimeter Walls

**Problem:** Contraband (drugs, phones, weapons) is thrown over prison walls.
Manual monitoring is impossible 24/7.

**Detector:** Structurally identical to LitterGuard's dump detector — an object
on a ballistic arc from outside the wall, landing inside. Differences:
(a) camera looks *up* at the wall-top, (b) the thrower is outside the frame,
(c) detection = projectile trajectory + landing. The track-birth +
stationarity logic (new object appears inside the wall where none existed)
works directly.

**Data:** No public dataset. Simulated throws (a ball tossed over a wall)
test the pipeline — same methodology as the wrapper-injection test.

**New work:** Camera-angle deployment + wall-line polygon.

**Ethics:** Detects *objects*, not people — lower privacy impact than
face-recognition. Evidence clip → guard dispatch. Not a surveillance dragnet.

---

## 4. Prisons — Aggression / Fall Detection in Common Areas

**Problem:** Fights and medical emergencies (falls, seizures) in prison yards
go undetected for minutes. Response time determines outcomes.

**Detector:** Pose-estimation (the `yolo11n-pose` channel, already built) on
yard cameras → abnormal pose patterns:
- Person horizontal on the ground >10 s (fall)
- Two persons with rapid, violent pose changes in close proximity (fight)
- Motionless person while others move away (potential medical event)

The temporal-logic engine applies: "person state NEW + STATIONARY +
horizontal" = fall alert.

**Data:** Public datasets exist (UR Fall, Le2i Fall) for pre-training;
prison-specific tuning needs one facility's cooperation.

**New work:** ~50 lines of pose-pattern rules.

**Ethics:** Protects inmates (medical response) rather than surveilling them.
Frame as safety monitoring with strict access controls and retention limits.

---

## 5. Border Forces — Illegal Crossing / Fence-Cut Detection

**Problem:** Border patrolling is manpower-intensive; incursions at night/fog
are missed. Ground sensors have gaps.

**Detector:** Fixed cameras or towers along the fence → person detection +
trajectory analysis (ByteTrack, already built) → a track that crosses the
fence-line polygon (`bin_zones` becomes `fence_zones` — same config field)
→ alert with direction. At night: thermal-camera input + the CLAHE profile.

**Data:** No public border dataset needed — the detector stack (person +
tracking + zone-crossing) requires no training; the geometry is the classifier.

**New work:** Fence polygon + direction logic (config only).

**Ethics:** Legitimate national-security perimeter monitoring — detecting
crossings at a defined boundary. Include human confirmation before any response.

---

## 6. Border Forces / Disaster — Vehicle Anomaly on Remote Roads

**Problem:** Remote border roads and disaster zones have no monitoring;
suspicious vehicle movement (smuggling, evacuation needs, unauthorized access)
goes undetected.

**Detector:** Vehicle tracking (already in `PersonVehicleChannel`) + zone
rules → vehicle stops in a restricted segment for >N minutes → alert; vehicle
travels wrong direction → alert; vehicle appears where none is expected
(time-of-day prior) → alert. All config-driven.

**Data:** Existing COCO vehicle classes + zone geometry from a map overlay.

**New work:** Zone/time rules (config only).

**Dual-use:** Disaster response (evacuation detection, stranded-vehicle
alerts) makes this fundable beyond security.

---

# Transferability Matrix

| Problem | Detector ensemble | Tracking | Temporal logic | Evidence packs | VLM verify | New work |
|---|---|---|---|---|---|---|
| Bin overflow | ✅ | — | inverted | ✅ | ✅ | ~20 lines |
| Encroachment | ✅ | ✅ | ✅ direct | ✅ | ✅ | zone polygons only |
| Contraband throw | ✅ | ✅ | ✅ direct | ✅ | ✅ | camera angle + wall-line |
| Aggression/fall | pose (✅ built) | ✅ | ✅ adapted | ✅ | ✅ | ~50 lines |
| Fence crossing | ✅ | ✅ | ✅ direct | ✅ | ✅ | fence polygon + direction |
| Vehicle anomaly | ✅ | ✅ | ✅ direct | ✅ | ✅ | zone/time rules |

---

# Recommended Build Order

1. **Bin overflow** — easiest, immediate civic value, no privacy issues
2. **Encroachment** — high political value in Indian municipalities
3. **Fence crossing** — border forces have budget; the tech is ready
4. **Contraband throws** — needs prison partnership; high impact
5. **Vehicle anomaly** — dual-use with disaster response
6. **Aggression/fall** — needs careful privacy design + one facility pilot

Each one is a repo + a pilot + a measurable outcome — the same methodology
proven on LitterGuard.

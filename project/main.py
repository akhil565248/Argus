"""
Auto-Find: Phase 1 — Core Detection & Tracking
================================================
Abandoned luggage / object detection using YOLOv8 + ByteTrack + a custom FSM.

Architecture:
  1. YOLOv8m runs inference and ByteTrack assigns persistent IDs.
  2. Detections are split into people (class 0) and bags (classes 24, 26, 27).
  3. AutoFindFSM consumes the raw detection tensors, manages owner-locking,
     handles ID-merge fallback for dropped bags, and drives the state machine
     (UNVERIFIED → IN_TRANSIT → WARNING → ABANDONED).
  4. When an item hits ABANDONED, crops of the item and last-known owner are
     saved to ./alerts/.

Usage:
  python main.py <path_to_video.mp4>
"""

import cv2
import json
import math
import os
import sys
from datetime import datetime
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
#  AutoFindFSM — Finite State Machine with ID-merge and owner-lock
# ─────────────────────────────────────────────────────────────────────────────

class AutoFindFSM:
    """
    Manages the lifecycle of every tracked bag/item through four states:

        UNVERIFIED  →  IN_TRANSIT  →  WARNING  →  ABANDONED

    Key features:
      • Owner-lock: once an owner is assigned, only THAT person's return
        resets the timer — random passers-by are ignored.
      • ID-merge: when ByteTrack assigns a new ID to a bag that hasn't moved
        (common when a worn backpack is dropped), the FSM transplants the old
        memory entry to the new ID so timers and owner data survive.
    """

    def __init__(self, distance_threshold=150, abandoned_frame_limit=150):
        self.bag_memory = {}                        # bag_id → dict
        self.distance_threshold = distance_threshold
        self.abandoned_frame_limit = abandoned_frame_limit

    # ── public entry-point (called once per frame) ──────────────────────────
    def update(self, detected_bags, detected_people):
        """
        Parameters
        ----------
        detected_bags  : list of per-detection box objects (from results[0].boxes)
                         that belong to bag classes.
        detected_people: list of per-detection box objects (from results[0].boxes)
                         that belong to class 0 (Person).

        Returns
        -------
        bag_memory : dict  — the full state dictionary keyed by bag track_id.
        """
        current_bag_ids = []

        # ── 1. Register / update every detected bag ────────────────────────
        for bag in detected_bags:
            bag_id = int(bag.id.item()) if bag.id is not None else None
            if bag_id is None:
                continue

            x1, y1, x2, y2 = bag.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_bag_ids.append(bag_id)

            if bag_id not in self.bag_memory:
                # ── ID-merge fallback ──────────────────────────────────────
                # If an old entry (now missing from the frame) sits within
                # 50 px of this "new" detection, assume it's the same bag
                # with a fresh tracker ID → transplant the old memory.
                merged = False
                for old_id, old_data in list(self.bag_memory.items()):
                    if old_id not in current_bag_ids:
                        old_cx, old_cy = old_data["coords"]
                        if math.dist((cx, cy), (old_cx, old_cy)) < 50:
                            self.bag_memory[bag_id] = self.bag_memory.pop(old_id)
                            self.bag_memory[bag_id]["coords"] = (cx, cy)
                            self.bag_memory[bag_id]["bounding_box"] = (x1, y1, x2, y2)
                            merged = True
                            break

                if not merged:
                    # Brand-new bag — start as UNVERIFIED
                    self.bag_memory[bag_id] = {
                        "owner_id": None,
                        "state": "UNVERIFIED",
                        "frames_alone": 0,
                        "coords": (cx, cy),
                        "bounding_box": (x1, y1, x2, y2),
                    }
            else:
                # Existing track — just refresh spatial data
                self.bag_memory[bag_id]["coords"] = (cx, cy)
                self.bag_memory[bag_id]["bounding_box"] = (x1, y1, x2, y2)

        # ── 2. Run state transitions ───────────────────────────────────────
        for bag_id, data in self.bag_memory.items():
            state = data["state"]

            # ── UNVERIFIED / IN_TRANSIT: find nearest person, lock owner ───
            if state in ["UNVERIFIED", "IN_TRANSIT"]:
                closest_person = None
                min_dist = float("inf")

                for person in detected_people:
                    if person.id is None:
                        continue
                    p_id = int(person.id.item())
                    px1, py1, px2, py2 = person.xyxy[0].tolist()
                    p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2

                    dist = math.dist(data["coords"], (p_cx, p_cy))
                    if dist < min_dist:
                        min_dist = dist
                        closest_person = p_id

                if min_dist < self.distance_threshold:
                    data["owner_id"] = closest_person
                    data["state"] = "IN_TRANSIT"
                    data["frames_alone"] = 0
                elif state == "IN_TRANSIT":
                    # Owner moved away → enter WARNING
                    data["state"] = "WARNING"

            # ── WARNING: only the locked owner can reset the timer ─────────
            elif state == "WARNING":
                owner_found = False
                owner_dist = float("inf")

                for person in detected_people:
                    if person.id is not None and int(person.id.item()) == data["owner_id"]:
                        owner_found = True
                        px1, py1, px2, py2 = person.xyxy[0].tolist()
                        p_cx, p_cy = (px1 + px2) / 2, (py1 + py2) / 2
                        owner_dist = math.dist(data["coords"], (p_cx, p_cy))
                        break

                if owner_found and owner_dist < self.distance_threshold:
                    data["state"] = "IN_TRANSIT"
                    data["frames_alone"] = 0
                else:
                    data["frames_alone"] += 1
                    if data["frames_alone"] >= self.abandoned_frame_limit:
                        data["state"] = "ABANDONED"

        return self.bag_memory


# ─────────────────────────────────────────────────────────────────────────────
#  Alert helper — crop and save item + owner on first ABANDONED trigger
# ─────────────────────────────────────────────────────────────────────────────

def trigger_alert(frame, bag_id, bag_data, detected_people, alerts_dir="alerts"):
    """Save crops of the abandoned item and the last-known owner to disk."""
    os.makedirs(alerts_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h, w = frame.shape[:2]

    # ── Item crop ──────────────────────────────────────────────────────────
    bx1, by1, bx2, by2 = [int(v) for v in bag_data["bounding_box"]]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w, bx2), min(h, by2)
    item_crop = frame[by1:by2, bx1:bx2]

    if item_crop.size > 0:
        path = os.path.join(alerts_dir, f"alert_{timestamp}_item_{bag_id}.jpg")
        cv2.imwrite(path, item_crop)
        print(f"[ALERT] Abandoned item detected! Saved crop → {path}")

    # ── Owner crop (scan current detections for the locked owner_id) ──────
    owner_id = bag_data.get("owner_id")
    if owner_id is not None:
        for person in detected_people:
            if person.id is not None and int(person.id.item()) == owner_id:
                px1, py1, px2, py2 = [int(v) for v in person.xyxy[0].tolist()]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)
                person_crop = frame[py1:py2, px1:px2]
                if person_crop.size > 0:
                    path = os.path.join(alerts_dir, f"alert_{timestamp}_person_{owner_id}.jpg")
                    cv2.imwrite(path, person_crop)
                    print(f"[ALERT] Associated person crop saved → {path}")
                break
    
    # ── Save Metadata for Phase 2 & 3 ──────────────────────────────────────
    metadata = {
        "timestamp": timestamp,
        "bag_id": bag_id,
        "owner_id": owner_id,
        "item_crop_path": os.path.join(alerts_dir, f"alert_{timestamp}_item_{bag_id}.jpg"),
        "status": "pending_vlm_review"
    }
    
    json_path = os.path.join(alerts_dir, f"alert_{timestamp}_metadata_{bag_id}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# BGR colour map
STATE_COLOURS = {
    "UNVERIFIED": (128, 128, 128),   # Grey  — not yet claimed
    "IN_TRANSIT": (255, 0, 0),       # Blue  — guarded
    "WARNING":    (0, 255, 255),     # Yellow — separated, timer running
    "ABANDONED":  (0, 0, 255),       # Red   — alert fired
}

COCO_NAMES = {24: "Backpack", 26: "Handbag", 27: "Umbrella"}


def draw_overlays(frame, tracked_bags, detected_people, detected_bags, fps):
    """Draw bounding boxes, labels, tethers, and person boxes on the frame."""

    # ── Person boxes (green) ───────────────────────────────────────────────
    for person in detected_people:
        if person.id is None:
            continue
        p_id = int(person.id.item())
        px1, py1, px2, py2 = [int(v) for v in person.xyxy[0].tolist()]
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {p_id}", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ── Bag boxes + state overlays ─────────────────────────────────────────
    for bag_id, data in tracked_bags.items():
        state = data["state"]

        # Skip unverified items (no owner provenance yet)
        if state == "UNVERIFIED":
            continue

        colour = STATE_COLOURS.get(state, (255, 255, 255))
        bx1, by1, bx2, by2 = [int(v) for v in data["bounding_box"]]
        cx, cy = int(data["coords"][0]), int(data["coords"][1])

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), colour, 2)

        # Label: class name + state + timer
        # Try to resolve the COCO class name from the raw detections
        class_label = "Item"
        for bag in detected_bags:
            if bag.id is not None and int(bag.id.item()) == bag_id:
                cls_id = int(bag.cls.item())
                class_label = COCO_NAMES.get(cls_id, f"Cls{cls_id}")
                break

        label = f"{class_label} {bag_id} [{state}]"
        if state in ["WARNING", "ABANDONED"] and fps > 0:
            secs = data["frames_alone"] // fps
            label += f" {secs}s"

        cv2.putText(frame, label, (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        # Tether line from item centroid → owner centroid while in WARNING
        if state == "WARNING" and data["owner_id"] is not None:
            for person in detected_people:
                if person.id is not None and int(person.id.item()) == data["owner_id"]:
                    px1, py1, px2, py2 = person.xyxy[0].tolist()
                    pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
                    cv2.line(frame, (cx, cy), (pcx, pcy), (0, 255, 255), 1)
                    break

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Main video loop
# ─────────────────────────────────────────────────────────────────────────────

def nothing(_):
    pass


def main():
    # ── Configuration ──────────────────────────────────────────────────────
    PERSON_CLASS = 0
    BAG_CLASSES = [24, 26, 27]                       # Backpack, Handbag, Umbrella
    ALL_CLASSES = [PERSON_CLASS] + BAG_CLASSES        # Feed all to YOLO
    DISTANCE_THRESHOLD = 150                          # px — owner proximity
    ABANDONED_SECONDS = 5                             # seconds until ABANDONED

    video_path = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading YOLOv8 medium model...")
    model = YOLO("yolov8m.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        print("Usage: python main.py <path_to_video.mp4>")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    abandoned_frame_limit = int(ABANDONED_SECONDS * fps)
    print(f"Video: {video_path} | {fps} FPS | {total_frames} frames")
    print(f"Abandonment threshold: {ABANDONED_SECONDS}s = {abandoned_frame_limit} frames")

    # ── Initialise FSM (exactly once) ──────────────────────────────────────
    #fsm = AutoFindFSM(
     #   distance_threshold=DISTANCE_THRESHOLD,
      #  abandoned_frame_limit=abandoned_frame_limit,
    #)

    fsm = AutoFindFSM(distance_threshold=150, abandoned_frame_limit=10)

    # Track which bag IDs have already fired an alert (one-shot)
    alerted_ids = set()

    # ── Window & trackbar ──────────────────────────────────────────────────
    window_name = "Auto-Find: Phase 1 (Core Tracking & Detection)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)
    cv2.createTrackbar("Timeline", window_name, 0, max(total_frames - 1, 1), nothing)

    paused = False
    print("Controls: SPACEBAR to Pause/Play, Q to Quit. Drag the trackbar to scrub.")

    # ── Frame loop ─────────────────────────────────────────────────────────
    while cap.isOpened():
        # Guard against window being closed manually
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            curr_trackbar_pos = cv2.getTrackbarPos("Timeline", window_name)
        except cv2.error:
            break

        current_video_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(current_video_frame - curr_trackbar_pos) > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_trackbar_pos)

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # ── YOLO inference + ByteTrack ─────────────────────────────────
            results = model.track(
                frame,
                classes=ALL_CLASSES,
                tracker="custom_bytetrack.yaml",
                imgsz=640,
                conf=0.15,
                iou=0.50,
                persist=True,
                verbose=False,
            )

            # ── Split detections into people vs bags ───────────────────────
            detected_people = []
            detected_bags = []

            if results[0].boxes.id is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    if cls_id == PERSON_CLASS:
                        detected_people.append(box)
                    elif cls_id in BAG_CLASSES:
                        detected_bags.append(box)

            # ── Feed into the FSM ──────────────────────────────────────────
            tracked_bags = fsm.update(detected_bags, detected_people)

            # ── Fire one-shot alerts for newly-ABANDONED items ─────────────
            for bag_id, data in tracked_bags.items():
                if data["state"] == "ABANDONED" and bag_id not in alerted_ids:
                    trigger_alert(frame, bag_id, data, detected_people)
                    alerted_ids.add(bag_id)

            # ── Draw everything ────────────────────────────────────────────
            annotated_frame = draw_overlays(
                frame, tracked_bags, detected_people, detected_bags, fps
            )

            # HUD banner
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Timeline", window_name, actual_pos)
            ts = actual_pos / fps if fps > 0 else 0
            status = f"Frame: {actual_pos}/{total_frames} | Time: {ts:.1f}s | PLAYING"
            cv2.putText(annotated_frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)
        else:
            cv2.setTrackbarPos("Timeline", window_name, current_video_frame)

        # ── Keyboard ───────────────────────────────────────────────────────
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")


if __name__ == "__main__":
    main()

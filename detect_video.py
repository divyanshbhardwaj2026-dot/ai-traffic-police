"""
detect_video.py — Vehicle Detection + Counting on Video / Live Camera
======================================================================
Usage:
    # Detect on a video file:
    python detect_video.py --source path/to/video.mp4

    # Use a live webcam (camera index 0):
    python detect_video.py --source 0

    # Adjust confidence and counting line position:
    python detect_video.py --source video.mp4 --confidence 0.4 --line 0.6

    # Save output video:
    python detect_video.py --source video.mp4 --save

What it does:
  1. Opens the video/camera stream frame by frame
  2. Runs YOLOv8 detection on each frame
  3. Uses ByteTrack (built into YOLOv8) to assign unique IDs to each vehicle
  4. Draws a virtual counting line across the frame
  5. Counts a vehicle only when it crosses the line (prevents double-counting)
  6. Displays annotated output in real-time
  7. Optionally saves the output video

Controls while running:
    Q  →  Quit
    P  →  Pause / Resume
    S  →  Save current frame as screenshot
"""

import argparse
import cv2
import os
import sys
import time
from collections import defaultdict
from ultralytics import YOLO
from utils import (
    VEHICLE_CLASSES,
    VEHICLE_COLORS,
    draw_box,
    draw_counts,
    draw_counting_line,
    has_crossed_line,
    save_results_csv,
    get_center,
)


def detect_video(
    source,
    confidence: float = 0.35,
    line_position: float = 0.5,
    save: bool = False,
    no_show: bool = False
):
    """
    Run vehicle detection and counting on a video or live camera.

    Args:
        source        : Path to video file, or integer camera index (0, 1, …)
        confidence    : Minimum detection confidence threshold
        line_position : Counting line Y position as fraction of frame height
                        (0.5 = middle, 0.6 = 60% down from top)
        save          : If True, save annotated video to output/
    """
    # ── 1. Load YOLOv8 model ──────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model (yolov8n.pt) …")
    model = YOLO("yolov8n.pt")   # Downloads automatically on first run (~6 MB)
    print("[INFO] Model loaded.")

    # ── 2. Open video source ──────────────────────────────────────────
    # Try converting source to int for webcam
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    # Get video properties
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    line_y   = int(frame_h * line_position)

    print(f"[INFO] Video source: {source}")
    print(f"[INFO] Resolution  : {frame_w}×{frame_h}  FPS: {fps:.1f}")
    print(f"[INFO] Counting line at Y = {line_y} ({line_position*100:.0f}% from top)")
    print("[INFO] Press  Q  to quit |  P  to pause |  S  to screenshot")

    # ── 3. Setup video writer (optional) ─────────────────────────────
    out_writer = None
    if save:
        os.makedirs("output", exist_ok=True)
        base_name  = os.path.splitext(os.path.basename(str(source)))[0]
        out_path   = f"output/{base_name}_counted.avi"
        fourcc     = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))
        print(f"[INFO] Saving output video → {out_path}")

    # ── 4. Tracking state ─────────────────────────────────────────────
    # Total crossed counts per vehicle class
    crossed_counts = defaultdict(int)

    # Previous Y positions per tracked vehicle ID
    # { track_id: (center_y, class_label) }
    prev_positions = {}

    # Set of IDs that have already been counted (to avoid double-counting)
    counted_ids = set()

    # ── 5. Main loop ──────────────────────────────────────────────────
    paused    = False
    frame_num = 0
    start_time = time.time()

    while True:
        # ── Pause handling ────────────────────────────────────────────
        if paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("p"):
                paused = False
            elif key == ord("q"):
                break
            continue

        # ── Read next frame ───────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_num += 1

        # ── Run YOLOv8 with ByteTrack tracker ─────────────────────────
        # persist=True tells YOLO to maintain tracker state across frames
        results = model.track(
            frame,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        # ── Draw counting line ────────────────────────────────────────
        draw_counting_line(frame, line_y)

        # ── Per-frame detection counts (for the count panel display) ──
        frame_counts = {"car": 0, "bus": 0, "motorcycle": 0, "truck": 0}

        # ── Process each detection ────────────────────────────────────
        if results.boxes is not None and results.boxes.id is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])

                # Skip non-vehicle detections
                if class_id not in VEHICLE_CLASSES:
                    continue

                label    = VEHICLE_CLASSES[class_id]
                conf     = float(box.conf[0])
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color    = VEHICLE_COLORS[label]
                cx, cy   = get_center(x1, y1, x2, y2)

                # ── Count crossing logic ──────────────────────────────
                if track_id in prev_positions:
                    prev_cy, _ = prev_positions[track_id]
                    if (track_id not in counted_ids and
                            has_crossed_line(cy, line_y, prev_cy)):
                        counted_ids.add(track_id)
                        crossed_counts[label] += 1

                # Update previous position
                prev_positions[track_id] = (cy, label)

                # Draw box with track ID
                id_label = f"{label}#{track_id}"
                draw_box(frame, x1, y1, x2, y2, id_label, conf, color)

                # Update frame display counts
                frame_counts[label] += 1

                # Draw small dot at vehicle center
                cv2.circle(frame, (cx, cy), 4, color, -1)

        # ── Build total crossed count for display ──────────────────────
        display_counts = {k: crossed_counts.get(k, 0) for k in frame_counts}
        total_crossed  = sum(display_counts.values())

        # ── Draw count panel ───────────────────────────────────────────
        draw_counts(frame, display_counts, total_crossed)

        # ── FPS overlay ───────────────────────────────────────────────
        elapsed = time.time() - start_time
        current_fps = frame_num / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame, f"FPS: {current_fps:.1f}",
            (frame_w - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255), 2, cv2.LINE_AA
        )

        # ── Write to output video ──────────────────────────────────────
        if out_writer:
            out_writer.write(frame)

        # ── Show the frame ─────────────────────────────────────────────
        if not no_show:
            cv2.imshow("AI Traffic Police — Video Detection", frame)
            # ── Keyboard controls ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = True
                print("[INFO] Paused. Press P to resume.")
            elif key == ord("s"):
                os.makedirs("output", exist_ok=True)
                screenshot_path = f"output/screenshot_frame{frame_num}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved → {screenshot_path}")

    # ── 6. Cleanup ─────────────────────────────────────────────────────
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

    # ── 7. Final results summary ───────────────────────────────────────
    final_counts = {k: crossed_counts.get(k, 0) for k in ["car", "bus", "motorcycle", "truck"]}
    final_total  = sum(final_counts.values())

    print("\n┌─────────────────────────────────┐")
    print("│    Final Vehicle Count Results  │")
    print("├─────────────────────────────────┤")
    for cls_name, count in final_counts.items():
        print(f"│  {cls_name:<14} : {count:<16}│")
    print("├─────────────────────────────────┤")
    print(f"│  TOTAL          : {final_total:<16}│")
    print("└─────────────────────────────────┘\n")

    if save:
        save_results_csv(final_counts, final_total, output_path="output/video_results.csv")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Traffic Police — Detect and count vehicles in video"
    )
    parser.add_argument(
        "--source", "-src",
        required=True,
        help="Path to video file OR camera index (e.g. 0 for webcam)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--line", "-l",
        type=float,
        default=0.5,
        help="Counting line Y position as fraction of frame height (default: 0.5)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save the annotated output video to the output/ folder"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display GUI window (useful for background processing)"
    )

    args = parser.parse_args()
    detect_video(args.source, args.confidence, args.line, args.save, args.no_show)

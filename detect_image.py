"""
detect_image.py — Vehicle Detection on a Single Image
======================================================
Usage:
    python detect_image.py --image path/to/image.jpg
    python detect_image.py --image path/to/image.jpg --confidence 0.4
    python detect_image.py --image path/to/image.jpg --save

What it does:
  1. Loads the image
  2. Runs YOLOv8 to detect all vehicles (car, bus, motorcycle, truck)
  3. Draws bounding boxes + labels
  4. Shows a count summary panel
  5. Displays the result (and optionally saves it)
"""

import argparse
import cv2
import os
import sys
from ultralytics import YOLO
from utils import (
    VEHICLE_CLASSES,
    VEHICLE_COLORS,
    draw_box,
    draw_counts,
    save_results_csv,
    get_center,
)


def detect_image(image_path: str, confidence: float = 0.25, save: bool = False, no_show: bool = False):
    """
    Run vehicle detection on a single image.

    Args:
        image_path : Path to the input image file
        confidence : Minimum confidence threshold (0.0 – 1.0)
        save       : If True, save the annotated image to output/
    """
    # ── 1. Validate input ─────────────────────────────────────────────
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # ── 2. Load YOLOv8 model ──────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model (yolov8n.pt) …")
    model = YOLO("yolov8n.pt")   # Downloads automatically on first run (~6 MB)
    print("[INFO] Model loaded.")

    # ── 3. Load the image ─────────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    print(f"[INFO] Image loaded: {image_path}  ({frame.shape[1]}×{frame.shape[0]} px)")

    # ── 4. Run inference ──────────────────────────────────────────────
    print(f"[INFO] Running detection (confidence threshold: {confidence}) …")
    results = model(frame, conf=confidence, verbose=False)[0]

    # ── 5. Process detections ─────────────────────────────────────────
    counts = {"car": 0, "bus": 0, "motorcycle": 0, "truck": 0}

    for box in results.boxes:
        class_id = int(box.cls[0])

        # Skip non-vehicle detections
        if class_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[class_id]
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = VEHICLE_COLORS[label]

        # Draw bounding box + label
        draw_box(frame, x1, y1, x2, y2, label, conf, color)

        # Increment class counter
        counts[label] += 1

    total = sum(counts.values())

    # ── 6. Draw count panel ───────────────────────────────────────────
    draw_counts(frame, counts, total)

    # ── 7. Print summary to terminal ──────────────────────────────────
    print("\n┌─────────────────────────────┐")
    print("│     Detection Results       │")
    print("├─────────────────────────────┤")
    for cls_name, count in counts.items():
        print(f"│  {cls_name:<12} : {count:<14}│")
    print("├─────────────────────────────┤")
    print(f"│  TOTAL        : {total:<14}│")
    print("└─────────────────────────────┘\n")

    # ── 8. Save output if requested ───────────────────────────────────
    if save:
        os.makedirs("output", exist_ok=True)
        base_name  = os.path.splitext(os.path.basename(image_path))[0]
        out_image  = f"output/{base_name}_detected.jpg"
        cv2.imwrite(out_image, frame)
        print(f"[INFO] Annotated image saved → {out_image}")
        save_results_csv(counts, total, output_path="output/image_results.csv")

    # ── 9. Display the result ─────────────────────────────────────────
    if not no_show:
        print("[INFO] Displaying result. Press any key to close …")
        cv2.imshow("AI Traffic Police — Image Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Traffic Police — Detect vehicles in a single image"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image (jpg, png, etc.)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save the annotated output image to the output/ folder"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display GUI window (useful in scripts/terminals)"
    )

    args = parser.parse_args()
    detect_image(args.image, args.confidence, args.save, args.no_show)


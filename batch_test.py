"""
batch_test.py — Batch Vehicle Detection & Evaluation on Dataset Images
========================================================================
Usage:
    python batch_test.py --dataset Vehicles-v1/valid --num 20
    python batch_test.py --dataset Vehicles-coco/test --num 50 --save

What it does:
  1. Picks N images from a dataset directory
  2. Runs YOLOv8 detection on each image
  3. Aggregates all vehicle counts
  4. Saves annotated images to output/batch_results/
  5. Exports a complete summary CSV report
"""

import argparse
import os
import glob
import pandas as pd
from ultralytics import YOLO
from utils import VEHICLE_CLASSES, VEHICLE_COLORS, draw_box, draw_counts, save_results_csv

def run_batch_test(dataset_dir: str, num_images: int = 20, confidence: float = 0.25, save: bool = True):
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory not found: {dataset_dir}")
        return

    # Find all jpg images
    image_files = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    if not image_files:
        print(f"[ERROR] No .jpg images found in {dataset_dir}")
        return

    selected_files = image_files[:num_images]
    print(f"[INFO] Found {len(image_files)} images. Testing first {len(selected_files)} images...")

    model = YOLO("yolov8n.pt")
    out_dir = "output/batch_results"
    if save:
        os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    total_aggregate = {"car": 0, "bus": 0, "motorcycle": 0, "truck": 0}

    for idx, img_path in enumerate(selected_files, 1):
        filename = os.path.basename(img_path)
        results = model(img_path, conf=confidence, verbose=False)[0]
        
        counts = {"car": 0, "bus": 0, "motorcycle": 0, "truck": 0}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                label = VEHICLE_CLASSES[cls_id]
                counts[label] += 1
                total_aggregate[label] += 1

        total_in_img = sum(counts.values())
        summary_rows.append({
            "image": filename,
            "car": counts["car"],
            "bus": counts["bus"],
            "motorcycle": counts["motorcycle"],
            "truck": counts["truck"],
            "total_vehicles": total_in_img
        })

        if save:
            # Save annotated image
            res_plotted = results.plot()
            import cv2
            cv2.imwrite(os.path.join(out_dir, f"detected_{filename}"), res_plotted)

        print(f"[{idx}/{len(selected_files)}] {filename[:30]}... -> Cars: {counts['car']}, Buses: {counts['bus']}, Bikes: {counts['motorcycle']}, Trucks: {counts['truck']}")

    df = pd.DataFrame(summary_rows)
    report_path = "output/batch_summary.csv"
    df.to_csv(report_path, index=False)
    print(f"\n[SUCCESS] Batch test complete!")
    print(f"[REPORT] Summary saved to: {report_path}")
    print("\nTotal Vehicles Detected Across Batch:")
    for k, v in total_aggregate.items():
        print(f"  - {k:<12}: {v}")
    print(f"  - TOTAL       : {sum(total_aggregate.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch test vehicle detection")
    parser.add_argument("--dataset", "-d", default="Vehicles-v1/valid", help="Path to images directory")
    parser.add_argument("--num", "-n", type=int, default=15, help="Number of images to test")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", "-s", action="store_true", default=True, help="Save annotated images")
    args = parser.parse_args()

    run_batch_test(args.dataset, args.num, args.confidence, args.save)

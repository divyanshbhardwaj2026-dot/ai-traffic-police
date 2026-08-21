"""
utils.py — Helper Functions for AI Traffic Police (Part 1)
----------------------------------------------------------
Contains all shared utilities:
  - Class label & color mapping
  - Bounding box drawing
  - On-screen results display
  - Virtual counting line logic
  - Results saving to CSV
"""

import cv2
import csv
import os
import datetime
import numpy as np

# ─────────────────────────────────────────────
# YOLO class IDs → Friendly vehicle labels
# These are the standard COCO class IDs that
# YOLOv8 uses for the 4 vehicle types we need
# ─────────────────────────────────────────────
VEHICLE_CLASSES = {
    2:  "car",
    5:  "bus",
    3:  "motorcycle",
    7:  "truck",
}

# Color for each vehicle type (B, G, R format for OpenCV)
VEHICLE_COLORS = {
    "car":        (0, 255, 0),      # Green
    "bus":        (255, 128, 0),    # Orange
    "motorcycle": (0, 200, 255),    # Yellow
    "truck":      (0, 0, 255),      # Red
}


def draw_box(frame, x1, y1, x2, y2, label, confidence, color):
    """
    Draw a single bounding box with a label on the frame.

    Args:
        frame       : The image/video frame (numpy array)
        x1,y1,x2,y2: Bounding box corner coordinates
        label       : Vehicle class name (e.g. 'car')
        confidence  : Detection confidence score (0.0 - 1.0)
        color       : (B, G, R) color tuple
    """
    # Draw the rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build the label text
    text = f"{label} {confidence:.0%}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size

    # Draw a filled background behind the text for readability
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)

    # Write the label text
    cv2.putText(
        frame, text,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255),  # White text
        2, cv2.LINE_AA
    )


def draw_counts(frame, counts, total):
    """
    Draw a summary count panel in the top-left corner of the frame.

    Args:
        frame  : The image/video frame
        counts : dict like {'car': 3, 'bus': 1, ...}
        total  : int, total number of vehicles
    """
    panel_h = 30 + (len(counts) + 1) * 28
    panel_w = 200

    # Semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "Vehicle Counts", (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Per-class counts
    y = 65
    for cls_name, count in counts.items():
        color = VEHICLE_COLORS.get(cls_name, (255, 255, 255))
        cv2.putText(frame, f"  {cls_name:<12}: {count}", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 28

    # Total
    cv2.putText(frame, f"  TOTAL       : {total}", (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_counting_line(frame, line_y, color=(0, 255, 255), thickness=2):
    """
    Draw the virtual counting line across the width of the frame.

    Args:
        frame     : The video frame
        line_y    : Y-coordinate (horizontal) of the counting line
        color     : Line color in BGR
        thickness : Line thickness in pixels
    """
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, thickness)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def has_crossed_line(center_y, line_y, prev_center_y):
    """
    Returns True if a vehicle center point has crossed the counting line
    from above to below (i.e., moving downward).

    Args:
        center_y      : Current Y center of the vehicle
        line_y        : Y position of the counting line
        prev_center_y : Y center from the previous frame
    Returns:
        bool
    """
    return prev_center_y < line_y <= center_y


def save_results_csv(counts, total, output_path="output/results.csv"):
    """
    Save the final vehicle count results to a CSV file.

    Args:
        counts      : dict like {'car': 3, 'bus': 1, ...}
        total       : int, total vehicles counted
        output_path : Path where the CSV will be written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Vehicle Type", "Count"])
        for cls_name, count in counts.items():
            writer.writerow([timestamp, cls_name, count])
        writer.writerow([timestamp, "TOTAL", total])

    print(f"[INFO] Results saved to: {output_path}")


def get_center(x1, y1, x2, y2):
    """Return the center (cx, cy) of a bounding box."""
    return (x1 + x2) // 2, (y1 + y2) // 2

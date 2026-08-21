# 🚦 AI Traffic Police (Part 1 - Foundations)

An AI-powered computer vision system designed to help traffic authorities monitor roads, detect incidents, and improve traffic management. This repository contains the complete implementation for **Part 1**, focusing on foundational vehicle classification and counting.

## 🎯 Features
- **Vehicle Classification**: Accurately detects and classifies 4 vehicle types (`car`, `bus`, `motorcycle`, `truck`) using a pre-trained YOLOv8 model.
- **Image Detection**: Processes static images to locate and tally vehicles.
- **Video & Live Tracking**: Processes video files or live webcam feeds. Integrates ByteTrack to assign unique IDs to vehicles, preventing double-counting.
- **Virtual Counting Line**: Configurable horizontal line that increments a counter only when a tracked vehicle crosses it.
- **Batch Processing**: Rapidly evaluates entire directories of images and outputs a summary CSV.
- **Interactive Web Dashboard**: Streamlit UI for easy drag-and-drop testing.

---

## 🛠️ Installation & Setup

**1. Clone the repository / Open the project folder**
Make sure you are in the `lucid-nobel` directory.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*(This includes `torch`, `opencv-python`, `ultralytics` (YOLOv8), `pandas`, and `streamlit`)*

---

## 🚀 How to Run

### 1. Interactive Web Dashboard (Streamlit)
The easiest way to test the system is via the web UI.
```bash
streamlit run app.py
```
This will open a browser window where you can upload images, adjust confidence thresholds, and instantly see the annotated results and count summaries.

### 2. Command Line - Single Image
Run detection on a single image and save the result to the `output/` folder:
```bash
python detect_image.py --image path/to/image.jpg --save
```

### 3. Command Line - Video / Live Camera
Run detection and counting on a video file. (Results are saved as `.avi` using the XVID codec for Windows compatibility).
```bash
python detect_video.py --source path/to/video.mp4 --save
```
*To use your webcam, pass `0` as the source:*
```bash
python detect_video.py --source 0
```

### 4. Command Line - Batch Testing
Test an entire folder of images (e.g., from the dataset) and generate a `batch_summary.csv` report:
```bash
python batch_test.py --dataset Vehicles-v1/valid --num 20
```

---

## 📁 Project Structure

```
├── app.py                 # Streamlit Web Dashboard
├── detect_image.py        # CLI script for single image detection
├── detect_video.py        # CLI script for video tracking & line-counting
├── batch_test.py          # CLI script for batch processing datasets
├── utils.py               # Shared helpers (drawing, CSV exports, math)
├── requirements.txt       # Python dependencies
├── output/                # Generated images, videos, and CSV reports
└── yolov8n.pt             # Pre-trained YOLO weights (auto-downloads)
```

---

## 📝 Part 1 Requirements Checklist
- [x] Classify vehicle types (car, bus, bike, truck)
- [x] Count vehicles from images/videos
- [x] Compulsory dataset analyzed and incorporated (used for batch testing & validation)

*Ready for Part 2!*

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from utils import VEHICLE_CLASSES, VEHICLE_COLORS, draw_box

# Page configuration
st.set_page_config(page_title="AI Traffic Police", page_icon="🚦", layout="wide")

st.title("🚦 AI Traffic Police Dashboard")
st.markdown("**Part 1 - Foundations:** Vehicle Classification & Counting")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    st.markdown("---")
    st.markdown("""
    **Supported Vehicles:**
    - 🚗 Car
    - 🚌 Bus
    - 🏍️ Motorcycle
    - 🚛 Truck
    """)

# Main UI
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png", "jfif", "webp"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    frame = np.array(image)
    
    # Handle RGB/BGR conversion
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame.copy()

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Run inference
        results = model(frame_bgr, conf=conf_threshold, verbose=False)[0]

        counts = {"car": 0, "bus": 0, "motorcycle": 0, "truck": 0}
        
        # Process bounding boxes
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    label = VEHICLE_CLASSES[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = VEHICLE_COLORS[label]
                    
                    # Draw box (BGR format)
                    draw_box(frame_bgr, x1, y1, x2, y2, label, conf, color)
                    counts[label] += 1

        # Convert back to RGB for Streamlit display
        annotated_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Detected Vehicles")
        st.image(annotated_img, use_container_width=True)

    # Statistics Section
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    total = sum(counts.values())
    
    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🚗 Cars", counts["car"])
    m2.metric("🚌 Buses", counts["bus"])
    m3.metric("🏍️ Motorcycles", counts["motorcycle"])
    m4.metric("🚛 Trucks", counts["truck"])
    m5.metric("✅ TOTAL", total)

    # Convert counts to a clean dataframe for display
    df = pd.DataFrame(list(counts.items()), columns=["Vehicle Type", "Count"])
    
    with st.expander("View Raw Data"):
        st.dataframe(df)

else:
    st.info("👈 Please upload an image from the sidebar or drag and drop one here to get started.")

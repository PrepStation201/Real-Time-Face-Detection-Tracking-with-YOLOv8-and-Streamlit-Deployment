import streamlit as st
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load models
model = YOLO("scr/yolov8n.pt")
tracker = DeepSort(max_age=30)

st.title("🔍 Real-Time Person Tracking with YOLOv8 + DeepSORT")
FRAME_WINDOW = st.image([])

# Use session state to persist `run` across reruns
if 'run' not in st.session_state:
    st.session_state.run = False

# Start & Stop buttons
col1, col2 = st.columns(2)
if col1.button("▶️ Start Webcam"):
    st.session_state.run = True
if col2.button("⛔ Stop Webcam"):
    st.session_state.run = False

# Webcam logic
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        # Only person detection (COCO class 0)
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

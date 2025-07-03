import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=30)

def run_tracking():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)[0]

        # Filter only 'person' class (class 0 in COCO)
        detections = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0:  # 0 = person
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (l + w, t + h), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLOv8 + DeepSort Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking()

# Real-time face detection
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # lightweight model

def run_face_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Face Detection (YOLOv8)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()

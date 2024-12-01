import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt
from sort import Sort

# Model path
model_path = 'basketball_yolo.pt'
model = YOLO(model_path)

# Initialize video capture
video_path = 'basketball.mp4'
cap = cv2.VideoCapture(video_path)

# Output settings
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'annotated_basketball_output.mp4')

# Video writer for annotated output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Variables for tracking performance
fps_values = deque(maxlen=50)
frame_id = 0
start_time = time.time()
accurate_detections = 0
total_detections = 0
tracker = Sort()

danger_detected = False

# Function to dynamically adjust confidence threshold
def get_dynamic_confidence(fps):
    if fps < 15:
        return 0.3
    elif fps < 25:
        return 0.5
    else:
        return 0.7

# Detect specific events (example: fouls)
def detect_fouls(detections, frame):
    foul_detected = False
    player_boxes = [det for det in detections if model.names[int(det.cls[0])] == 'player']

    for i, player1 in enumerate(player_boxes):
        for player2 in player_boxes[i+1:]:
            if overlap(player1, player2):
                cv2.putText(frame, "Foul Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                foul_detected = True
    return foul_detected

# Overlap function for foul detection
def overlap(box1, box2):
    x1, y1, x2, y2 = map(int, box1.xyxy[0])
    px1, py1, px2, py2 = map(int, box2.xyxy[0])
    return not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    elapsed_time = time.time() - start_time

    # Object detection with YOLO
    results = model(frame)
    detections = results[0].boxes
    detections_array = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]

        dynamic_conf_threshold = get_dynamic_confidence(frame_id / elapsed_time)
        if conf > dynamic_conf_threshold:
            accurate_detections += 1
        total_detections += 1

        detections_array.append([x1, y1, x2, y2, conf])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update SORT tracker
    detections_array = np.array(detections_array)
    if detections_array.size > 0:
        tracked_objects = tracker.update(detections_array)
        for tracked in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, tracked)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Event detection
    danger_detected = detect_fouls(detections, frame) or danger_detected

    # Real-time FPS plot
    current_fps = frame_id / elapsed_time
    fps_values.append(current_fps)
    if len(fps_values) > 1:
        plt.plot(range(len(fps_values)), fps_values, 'r-')
        plt.pause(0.01)

    # Save annotated frame to video
    out.write(frame)
    cv2.imshow("Enhanced YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Final performance report
accuracy = (accurate_detections / total_detections) * 100 if total_detections > 0 else 0
avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

report_path = os.path.join(output_dir, 'evaluation_report.txt')
with open(report_path, 'w') as report_file:
    report_file.write("Performance Evaluation Report\n")
    report_file.write(f"Average FPS: {avg_fps:.2f}\n")
    report_file.write(f"Detection Accuracy: {accuracy:.2f}%\n")
    report_file.write(f"Dangerous Situation Detected: {'Yes' if danger_detected else 'No'}\n")

plt.figure(figsize=(10, 5))
plt.plot(range(len(fps_values)), fps_values, 'r-')
plt.title('FPS over Frames')
plt.xlabel('Frame')
plt.ylabel('FPS')
plt.axhline(y=avg_fps, color='b', linestyle='--', label=f'Avg FPS: {avg_fps:.2f}')
plt.legend()
plt.grid()
plt.show()

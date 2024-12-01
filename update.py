import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
import matplotlib.pyplot as plt
from collections import deque

# Initialize YOLO model
model_path = 'basketball_yolo.pt'  # Ensure the path to the trained YOLO model
model = YOLO(model_path)

# Initialize video capture
video_path = 'basketball.mp4'  # Your video path
cap = cv2.VideoCapture(video_path)

# Output settings
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, 'motion_tracking_output.mp4')

# Video writer for annotated output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Variables for tracking performance
fps_values = deque(maxlen=50)
frame_id = 0
start_time = time.time()
player_positions = {}

# Tracker
tracker = Sort()

# Court boundaries for Out-of-Bounds check
court_boundaries = {
    'left': 100,
    'right': frame_width - 100,
    'top': 100,
    'bottom': frame_height - 100
}

# Function to calculate FPS dynamically
def calculate_fps():
    elapsed_time = time.time() - start_time
    current_fps = frame_id / elapsed_time if elapsed_time > 0 else fps
    fps_values.append(current_fps)
    return current_fps

# Function to check for violations (fouls, double dribbling, out of bounds)
def check_violations(player_id, bbox, player_positions, ball_position):
    x1, y1, x2, y2 = bbox
    fouls = False
    out_of_bounds = False

    # Out of bounds check (on court boundaries)
    if x1 < court_boundaries['left'] or x2 > court_boundaries['right'] or y1 < court_boundaries['top'] or y2 > court_boundaries['bottom']:
        out_of_bounds = True

    # Foul detection (simple overlap check)
    for other_id, other_position in player_positions.items():
        if other_id != player_id:
            ox1, oy1, ox2, oy2 = other_position
            if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):  # simple AABB collision check
                fouls = True

    return fouls, out_of_bounds

# Function to draw bounding box, text, and step count
def draw_bounding_box_and_text(frame, x1, y1, x2, y2, label, conf, player_id=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {conf:.2f}"
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Separate figure for FPS
plt.ion()
fig_fps, ax_fps = plt.subplots()
ax_fps.set_title('FPS')
ax_fps.set_xlabel('Frames')
ax_fps.set_ylabel('FPS')

# Ball tracking (placeholder for ball detection logic)
ball_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    current_fps = calculate_fps()

    # YOLO detection
    results = model(frame)
    detections = results[0].boxes
    detections_array = []

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]

        if conf > 0.5:  # Filtering based on confidence threshold
            detections_array.append([x1, y1, x2, y2, conf])
            draw_bounding_box_and_text(frame, x1, y1, x2, y2, label, conf)

    trackers = tracker.update(np.array(detections_array))

    # Update FPS plot
    ax_fps.plot(range(len(fps_values)), fps_values, 'g')
    fig_fps.canvas.draw()
    fig_fps.canvas.flush_events()

    # Write the frame to the output video
    out.write(frame)

    # Display the video on the screen
    cv2.imshow('Tracking & Performance', frame)

    # Close when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Evaluation report generation
avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

report_path = os.path.join(output_dir, 'evaluation_report.txt')
with open(report_path, 'w') as report_file:
    report_file.write("Performance Evaluation Report\n")
    report_file.write(f"Average FPS: {avg_fps:.2f}\n")

cap.release()
out.release()
cv2.destroyAllWindows()

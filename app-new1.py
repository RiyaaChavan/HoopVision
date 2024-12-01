
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
from collections import deque
import matplotlib.pyplot as plt
import threading
import queue

# Initialize YOLO model
model_path = 'basketball_yolo.pt'
model = YOLO(model_path)

# Initialize video capture
video_path = 'dribblenba.mp4'
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
accuracy_values = deque(maxlen=50)
frame_id = 0
start_time = time.time()
player_positions = {}
step_count = {}

# Tracker
tracker = Sort()

# Initialize centroid tracking
player_centroids = {}

# Create a queue for FPS values to be plotted
fps_queue = queue.Queue()

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to calculate steps (distance traveled)
def calculate_steps(player_id, prev_position, curr_position):
    if player_id not in player_centroids:
        player_centroids[player_id] = []
    
    player_centroids[player_id].append(curr_position)
    if len(player_centroids[player_id]) > 1:
        # Calculate Euclidean distance between last and current position
        steps = euclidean_distance(player_centroids[player_id][-2], curr_position)
        if steps > 30:  # Threshold for steps (can adjust)
            if player_id not in step_count:
                step_count[player_id] = 0
            step_count[player_id] += 1

# Function to draw bounding box and display counters
def draw_bounding_box_and_text(frame, x1, y1, x2, y2, label, conf, player_id=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {conf:.2f}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if player_id is not None:
        step_text = f"Steps: {step_count.get(player_id, 0)}"
        cv2.putText(frame, step_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Calculate FPS dynamically and add to the frame
def calculate_fps():
    elapsed_time = time.time() - start_time
    current_fps = frame_id / elapsed_time if elapsed_time > 0 else fps
    fps_values.append(current_fps)
    return current_fps

# Function to detect out of bounds violation
def detect_out_of_bounds(player_positions, frame, frame_width, frame_height):
    for player_id, position in player_positions.items():
        if position[0] < 0 or position[0] > frame_width or position[1] < 0 or position[1] > frame_height:
            cv2.putText(frame, f"Player {player_id} Out of Bounds", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Function to detect double dribble violation
def detect_double_dribble(ball_position, ball_previous_position, frame):
    if ball_position and ball_previous_position:
        if np.linalg.norm(np.array(ball_position) - np.array(ball_previous_position)) == 0:
            cv2.putText(frame, "Double Dribbling", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Function to detect foul based on player proximity
def detect_foul(tracked_objects, frame):
    event_detected = None
    for i, obj1 in enumerate(tracked_objects):
        for obj2 in tracked_objects[i+1:]:
            x1, y1, _, _, _ = obj1
            x2, y2, _, _, _ = obj2
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            if distance < 50:  # Threshold for player proximity (foul detection)
                event_detected = "Out of Bounds Detected"
                cv2.putText(frame, event_detected, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return event_detected

# Detect and display violations during video processing
def process_violations(frame, tracked_objects, ball_position, ball_previous_position, player_positions, frame_width, frame_height):
    detect_out_of_bounds(player_positions, frame, frame_width, frame_height)
    detect_double_dribble(ball_position, ball_previous_position, frame)
    detect_foul(tracked_objects, frame)

# FPS and Accuracy plotting function in background
def plot_fps_accuracy():
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title("FPS over Time", fontsize=14)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.set_ylim(0, 60)
    line1, = ax1.plot([], [], label='FPS', color='blue')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Accuracy over Time", fontsize=14)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    line2, = ax2.plot([], [], label='Accuracy', color='green')
    ax2.legend()
    ax2.grid(True)

    frame_counter = 0
    while True:
        try:
            current_fps = fps_queue.get(timeout=1)  # Get FPS value from the queue
            accuracy = 100  # Replace with actual accuracy calculation logic
            accuracy_values.append(accuracy)
            fps_values.append(current_fps)

            frame_counter += 1
            line1.set_xdata(np.arange(frame_counter))
            line1.set_ydata(fps_values)
            line2.set_xdata(np.arange(frame_counter))
            line2.set_ydata(accuracy_values)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            plt.pause(0.1)
        except queue.Empty:
            break

# Start FPS and Accuracy plotting thread
fps_thread = threading.Thread(target=plot_fps_accuracy, daemon=True)
fps_thread.start()

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

    ball_position = None
    ball_previous_position = None

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]

        if conf > 0.5:  # Filtering based on confidence threshold
            detections_array.append([x1, y1, x2, y2, conf])
            draw_bounding_box_and_text(frame, x1, y1, x2, y2, label, conf)

            if label == 'ball':
                ball_previous_position = ball_position
                ball_position = (x1, y1)

    # Update tracker
    trackers = tracker.update(np.array(detections_array))

    # Process and display events
    process_violations(frame, trackers, ball_position, ball_previous_position, player_positions, frame_width, frame_height)

    # Write annotated frame to output video
    out.write(frame)

    # Show the frame
    cv2.imshow('Basketball Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


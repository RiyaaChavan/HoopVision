import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
import matplotlib.pyplot as plt
import streamlit as st
from collections import deque

# Initialize YOLO model
model_path = 'basketball_yolo.pt'  # Path to your trained YOLO model
video_path = 'basketball.mp4'      # Path to your video file

# Check if the model file and video file exist
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
elif not os.path.exists(video_path):
    st.error(f"Video file not found at {video_path}")
else:
    model = YOLO(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video file at {video_path}")
    else:
        # Streamlit UI setup
        st.title("Basketball Motion Tracking and Violation Detection")
        st.sidebar.header("Settings")
        output_video_path = os.path.join('./outputs', 'motion_tracking_output.mp4')
        
        # Output directory setup
        output_dir = './outputs'
        os.makedirs(output_dir, exist_ok=True)

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
        def draw_bounding_box_and_text(frame, x1, y1, x2, y2, label, conf):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ball tracking
        ball_position = None
        ball_previous_position = None

        # Processing video frames
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

                    if label == 'ball':
                        ball_previous_position = ball_position
                        ball_position = (x1, y1)

            # Update tracker
            trackers = tracker.update(np.array(detections_array))

            # Detect violations
            for track in trackers:
                x1, y1, x2, y2, track_id = track.astype(int)
                player_positions[track_id] = (x1, y1, x2, y2)
                fouls, out_of_bounds = check_violations(track_id, (x1, y1, x2, y2), player_positions, ball_position)

                # Display violations
                if fouls:
                    cv2.putText(frame, "Foul Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if out_of_bounds:
                    cv2.putText(frame, "Out of Bounds", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write annotated frame to output video
            out.write(frame)
            st.image(frame, channels="BGR", caption="Tracking & Violations in Real-time")

            # Stop if 'q' is pressed (for local testing; Streamlit has no key listener)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        out.release()
        st.success("Video processing completed. Check the outputs folder for the annotated video.")
        
        # Display the saved video
        st.video(output_video_path)

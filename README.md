
# HoopVision - Basketball Action and Object Detection with YOLOv8

## Project Overview

This project implements a computer vision system to detect objects (players, ball), track actions, and classify events within basketball videos. Utilizing the powerful YOLOv8 model for object detection and the SORT algorithm for tracking, this system provides insights into player movement, ball possession, and identifies potential violations like fouls or out-of-bounds plays. The processed video is then outputted with bounding boxes, labels, and real-time performance metrics are monitored.

## Features

* **Object Detection:** Detects basketball players and the basketball using a custom-trained YOLOv8 model.
* **Object Tracking:** Employs the SORT (Simple Online and Realtime Tracking) algorithm to maintain consistent IDs for detected objects across frames, enabling robust tracking of players and the ball.
* **Action Classification (Implicit):** While not explicitly classifying high-level actions (e.g., "shooting", "dribbling"), the tracking of player and ball positions provides foundational data for such analyses.
* **Violation Detection (Basic):** Includes rudimentary checks for:
    * **Out-of-Bounds:** Detects if players or the ball move beyond predefined court boundaries.
    * **Foul Detection:** A basic collision detection between player bounding boxes suggests potential fouls.
* **Real-time Performance Monitoring:** Displays and logs the Frames Per Second (FPS) of the processing pipeline, providing insight into the system's efficiency.
* **Annotated Video Output:** Generates an output video with detected objects, their labels, confidence scores, and tracking IDs.
* **Performance Report:** Generates a text file summarizing key performance metrics like average FPS.

## Technical Details

### YOLOv8 Model

The core of the object detection is powered by `YOLOv8`, a state-of-the-art object detection model known for its speed and accuracy. A custom `basketball_yolo.pt` model is loaded, indicating that the model has been trained specifically on basketball-related datasets to accurately identify players and the ball.

### SORT Tracker

For robust object tracking, the `SORT` algorithm is integrated. SORT is a pragmatic and effective tracking-by-detection framework that associates new detections with existing tracks based on their predicted positions and appearances. This ensures that each detected player and the ball maintains a consistent ID throughout the video sequence.

### Violation Logic

* **Out-of-Bounds:** The code defines `court_boundaries` (left, right, top, bottom). Any detected player or ball bounding box extending beyond these coordinates is flagged as out-of-bounds.
* **Foul Detection:** A simple Axis-Aligned Bounding Box (AABB) collision detection is performed between player bounding boxes. If two player bounding boxes overlap, it's flagged as a potential foul. *Note: This is a very basic foul detection and would require more sophisticated logic (e.g., motion vectors, rules of basketball) for a robust system.*

### Performance Monitoring

The `calculate_fps` function dynamically computes the frames per second by tracking the number of processed frames and elapsed time. A deque `fps_values` stores recent FPS values to smooth out fluctuations and provide a more stable average. A real-time FPS plot is also displayed using `matplotlib`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install opencv-python ultralytics numpy matplotlib
    pip install git+https://github.com/abewley/sort.git # For the SORT tracker
    ```
    *Ensure you have a custom-trained YOLOv8 model named `basketball_yolo.pt` in your project directory.*

## Usage

1.  **Place your video:**
    Ensure your basketball video file (e.g., `basketball.mp4`) is in the same directory as the script, or update the `video_path` variable in the code.

2.  **Place your YOLOv8 model:**
    Make sure your trained YOLOv8 model (`basketball_yolo.pt`) is in the same directory, or update the `model_path` variable.

3.  **Run the script:**
    ```bash
    python your_script_name.py
    ```

    The script will:
    * Open a window displaying the annotated video with detections and tracking.
    * Open a separate `matplotlib` window showing the real-time FPS.
    * Save the processed video to the `outputs` directory as `motion_tracking_output.mp4`.
    * Generate a `evaluation_report.txt` in the `outputs` directory with performance metrics.

4.  **Quit:**
    Press `q` on the video display window to stop the processing and close all windows.

## Project Structure

```
.
├── basketball_yolo.pt        # Trained YOLOv8 model weights
├── basketball.mp4            # Input basketball video
├── your_script_name.py       # Main Python script
├── outputs/                  # Directory for output files
│   ├── motion_tracking_output.mp4 # Annotated output video
│   └── evaluation_report.txt    # Performance report
└── README.md                 # This README file
```

## Future Enhancements

* **Advanced Action Recognition:** Implement more sophisticated machine learning models (e.g., LSTMs, Transformers) to recognize complex basketball actions like shooting, dribbling, passing, and defending.
* **Detailed Violation Analysis:** Integrate more rules-based logic for basketball violations, such as travel, double dribble, shot clock violations, etc. This would require more precise trajectory analysis and event sequencing.
* **Team Classification:** Differentiate between players from opposing teams.
* **Player Statistics:** Track individual player statistics like possession time, number of passes, shots attempted, etc.
* **3D Pose Estimation:** Incorporate 3D pose estimation to better understand player body language and interactions.
* **UI/Dashboard:** Develop a graphical user interface (GUI) or web dashboard for easier interaction, visualization, and analysis of results.
* **Real-time Stream Processing:** Adapt the code to process live video streams from cameras.

---

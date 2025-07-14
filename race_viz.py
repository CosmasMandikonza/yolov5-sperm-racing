"""
race_viz.py.
-----------
Visualizes detections and tracks of sperm cells using YOLOv5 and SORT tracker.

Written by: CosmasMandikonza]
Date: [2025-05-23]

Approach:
---------
1. Run YOLOv5 object detection on the cropped input video to detect sperm heads.
2. Use the SORT algorithm for tracking individual sperm cells across frames.
3. Visualize and save the output video with tracking IDs and bounding boxes.

Assumptions:
- The YOLOv5 weights are trained for this dataset and located at the provided path.
- Input video is pre-cropped to match training/annotation region.
- All dependencies are installed and available.
"""

import cv2
import numpy as np

from sort import Sort  # Make sure you have sort.py in the same directory

# ----------- PARAMETERS (edit as needed) -------------------
INPUT_VIDEO = "cropped_video.avi"
YOLO_WEIGHTS = "spermtrack/exp2/weights/best.pt"
IMG_SIZE = 640  # YOLOv5 input image size
CONF_THRESHOLD = 0.25  # Detection confidence threshold

OUTPUT_VIDEO = "output_leader_viz.avi"  # Output file name

# -----------------------------------------------------------

# Load YOLOv5 model (using torch.hub or ultralytics, adjust for your setup)
import torch

model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=YOLO_WEIGHTS, source="local"
)  # source='local' for local weights

# Initialize video reader and writer
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# Initialize the SORT tracker
tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.2)

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. Run YOLOv5 detection ---
    # Convert BGR to RGB for YOLOv5
    results = model(frame[..., ::-1], size=IMG_SIZE)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    # Prepare detections for SORT: [x1, y1, x2, y2, score]
    dets_for_sort = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf >= CONF_THRESHOLD:
            dets_for_sort.append([x1, y1, x2, y2, conf])

    dets_for_sort = np.array(dets_for_sort)

    # --- 2. Run SORT tracking ---
    tracked_objects = tracker.update(dets_for_sort)

    # --- 3. Visualization ---
    for d in tracked_objects:
        x1, y1, x2, y2, track_id = d
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        # Draw rectangle and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Optional: Write frame number on video for debug
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    frame_num += 1

    # (Optional) to show in real-time
    # cv2.imshow('Tracking', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved output video with tracking: {OUTPUT_VIDEO}")

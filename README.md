# Sperm Racing AI Detection

A YOLOv5-based pipeline for robust sperm detection and tracking in microscope video frames.

## Approach

- **Tried and discarded approaches:** I explored using pre-trained models (Cellpose, DeepSort, etc.), but found YOLOv5 provided the best performance and was fastest to adapt for my dataset, which was limited in both number and variety.
- **Preprocessing:** Used `crop.py` to crop videos to the region where sperm appear (coordinates: x=2558, y=1466, width=1240, height=446), ensuring that both training data and inference were consistent.
- **Model Training:** Labeled ~50 images in Roboflow, exported in YOLOv5 format, and trained for 100 epochs using `train.py`.
- **Detection:** Used `detect.py` to run inference on both still frames and the cropped video, outputting bounding boxes and confidence scores.
- **Visualization:** Wrote `race_viz.py` to visualize detections and can be adapted for tracking/race logic if required.
- **Time constraints:** The solution balances accuracy and reproducibility under the short project deadline.

## Usage

### Setup

```bash
git clone https://github.com/yourusername/yolov5-sperm-racing.git
cd yolov5-sperm-racing
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

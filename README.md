# 🎯 CodeAlpha — Object Detection & Tracking (Task 4)

Real-time object detection and multi-object tracking built for the **CodeAlpha Artificial Intelligence Internship**. Detects objects in video/webcam streams with **YOLOv8** and tracks them across frames with **ByteTrack** (a SORT-family tracking algorithm), assigning each object a persistent ID.

## ✨ Features

- **YOLOv8** pre-trained detection (80 COCO classes: person, car, bicycle, dog, ...)
- **ByteTrack / BoT-SORT** multi-object tracking with **persistent track IDs**
- Works with **video files** and **live webcam**
- Color-coded bounding boxes + labels with class name & confidence
- **Motion trails** showing each object's recent path
- Live **FPS counter** and unique-object count HUD
- Option to **save the annotated output video**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8n (Ultralytics) |
| Tracking | ByteTrack / BoT-SORT |
| Video I/O & Display | OpenCV |

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on the sample video
python detect_track.py

# Run on any video
python detect_track.py --source videos/car-detection.mp4

# Run on webcam
python detect_track.py --source 0

# Save annotated output video (no window)
python detect_track.py --save --no-show

# Use BoT-SORT tracker instead of ByteTrack
python detect_track.py --tracker botsort.yaml
```

Press **`q`** to quit the video window.

## 📸 Screenshots & Results

![Detection output](screenshots/detection.png)

Test run on the sample video: **647 frames processed, 11 unique objects tracked, ~17 FPS** on CPU.

## 📖 How It Works

1. OpenCV reads frames from the video file or webcam.
2. Each frame is passed to YOLOv8, which detects objects and returns bounding boxes, classes, and confidence scores.
3. The ByteTrack algorithm associates detections across frames, assigning a **persistent ID** to every object.
4. Boxes, labels, IDs, motion trails, and a live FPS counter are drawn on each frame in real time.

## 📂 Project Structure

```
CodeAlpha_ObjectDetectionTracking/
├── detect_track.py    # Main script (detection + tracking)
├── yolov8n.pt         # Pre-trained YOLOv8-nano weights
├── videos/            # Input test videos
├── output/            # Saved annotated videos (with --save)
└── requirements.txt
```

---

*CodeAlpha AI Internship — Task 4 | Repository: `CodeAlpha_ObjectDetectionTracking`*

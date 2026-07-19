"""
CodeAlpha Internship - Task 4: Object Detection and Tracking
------------------------------------------------------------
Real-time object detection + multi-object tracking using:
- YOLOv8 (Ultralytics) for detection
- ByteTrack (SORT-family algorithm) for tracking with persistent IDs
- OpenCV for video I/O and display

Each detected object gets a bounding box, class label, confidence score and
a persistent tracking ID that follows it across frames.

Usage:
    # Video file (default: videos/person-bicycle-car-detection.mp4)
    python detect_track.py

    # Any video file
    python detect_track.py --source videos/car-detection.mp4

    # Webcam
    python detect_track.py --source 0

    # Save the annotated output video, without opening a display window
    python detect_track.py --save --no-show
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DEFAULT_SOURCE = BASE_DIR / "videos" / "person-bicycle-car-detection.mp4"
DEFAULT_MODEL = BASE_DIR / "yolov8n.pt"
OUTPUT_DIR = BASE_DIR / "output"

# A fixed color palette so each track ID keeps a consistent color
PALETTE = [
    (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
    (238, 130, 238), (0, 206, 209), (255, 140, 0), (144, 238, 144),
    (219, 112, 147), (100, 149, 237),
]


def color_for_id(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack object detection and tracking"
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="video file path, or 0 for webcam")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="path to YOLOv8 weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="detection confidence threshold")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="tracking algorithm config")
    parser.add_argument("--save", action="store_true",
                        help="save annotated output video to output/")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open a display window (useful on servers)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Webcam if the source is a digit ("0", "1", ...)
    source = int(args.source) if str(args.source).isdigit() else args.source

    print(f"📦 Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        name = "webcam" if isinstance(source, int) else Path(args.source).stem
        out_path = OUTPUT_DIR / f"{name}_tracked.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in,
            (width, height),
        )
        print(f"💾 Saving output to: {out_path}")

    # Keep a short motion trail for every track ID
    trails = defaultdict(list)
    unique_ids = set()
    frame_count = 0
    start_time = time.time()

    print("🎬 Processing... (press 'q' to quit)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Detection + tracking in one call (persist keeps IDs across frames)
        results = model.track(
            frame,
            conf=args.conf,
            tracker=args.tracker,
            persist=True,
            verbose=False,
        )
        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), track_id, cls_id, conf in zip(
                boxes, ids, classes, confs
            ):
                unique_ids.add(track_id)
                color = color_for_id(track_id)
                label = f"ID {track_id} | {model.names[cls_id]} {conf:.2f}"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with filled background
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                )

                # Motion trail (center points of the last 30 frames)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trails[track_id].append((cx, cy))
                trails[track_id] = trails[track_id][-30:]
                for i in range(1, len(trails[track_id])):
                    cv2.line(
                        frame,
                        trails[track_id][i - 1],
                        trails[track_id][i],
                        color,
                        2,
                    )

        # HUD: FPS + object count
        elapsed = time.time() - start_time
        fps_live = frame_count / elapsed if elapsed > 0 else 0
        hud = f"FPS: {fps_live:.1f} | Objects tracked: {len(unique_ids)}"
        cv2.rectangle(frame, (5, 5), (360, 35), (0, 0, 0), -1)
        cv2.putText(
            frame, hud, (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        if writer is not None:
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n✅ Done!")
    print(f"   Frames processed:      {frame_count}")
    print(f"   Unique objects tracked: {len(unique_ids)}")
    print(f"   Average FPS:            {fps_live:.1f}")


if __name__ == "__main__":
    main()

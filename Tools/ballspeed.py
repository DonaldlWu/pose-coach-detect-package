import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "input")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
MODEL_PATH = os.path.join(REPO_ROOT, "yolov8n.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python3 Tools/ballspeed.py <video_filename>")
    print(f"  Place video files in '{INPUT_DIR}'.")
    sys.exit(1)

input_name = sys.argv[1]
fname = os.path.join(INPUT_DIR, input_name)
if not os.path.exists(fname):
    print(f"Error: '{fname}' not found.")
    sys.exit(1)

stem = os.path.splitext(input_name)[0]
ext = os.path.splitext(input_name)[1]
output_fname = os.path.join(OUTPUT_DIR, f"{stem}_output{ext}")

CONF_THRESHOLD = 0.45   # minimum YOLO confidence to accept detection
MAX_TRAIL_JUMP = 300    # reset trail if detection jumps further than this (px)

SPORTS_BALL_CLASS = 32     # COCO class id for sports ball

# =========================================================================

model = YOLO(MODEL_PATH)


def track_baseball(video_path):
    cap = cv2.VideoCapture(video_path)

    # obtain video info ------------------------------------------------
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f'Video resolution: {width} x {height}, frame rate: {fps} fps, frame count: {frame_count}.')

    # obtain video frames ----------------------------------------------
    trail = []
    frames = []
    result_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # detect ball from frames ----------------------------------------
    for frame in tqdm(frames):
        result_frame = frame.copy()

        # run YOLO inference (sports ball class only)
        results = model(
            frame, classes=[SPORTS_BALL_CLASS], conf=CONF_THRESHOLD, verbose=False)

        # pick highest-confidence detection
        best_box = None
        best_conf = 0.0
        for box in results[0].boxes:
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_box = box

        # update trail
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            center = (cx, cy)
            if trail and np.hypot(cx - trail[-1][0], cy - trail[-1][1]) > MAX_TRAIL_JUMP:
                trail.clear()
            trail.append(center)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_frame, f"{best_conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for j in range(1, len(trail)):
            cv2.line(result_frame, trail[j-1], trail[j], (255, 0, 0), 3)
        for pt in trail:
            cv2.circle(result_frame, pt, 3, (0, 255, 255), -1)

        result_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))

    return result_frames, trail, width, height, fps


result_frames, trail, width, height, fps = track_baseball(fname)


def save_video(frames, fps, output_path="output.mp4"):
    if not frames:
        print("No frames to save.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved to {os.path.abspath(output_path)}")


save_video(result_frames, fps, output_fname)

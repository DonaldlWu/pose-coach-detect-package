import cv2
import numpy as np
from tqdm import tqdm
import os
import sys

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python3 ballspeed.py <video_filename>")
    print(f"  Place video files in the '{INPUT_DIR}/' folder.")
    sys.exit(1)

input_name = sys.argv[1]
fname = os.path.join(INPUT_DIR, input_name)
if not os.path.exists(fname):
    print(f"Error: '{fname}' not found.")
    sys.exit(1)

stem = os.path.splitext(input_name)[0]
ext  = os.path.splitext(input_name)[1]
output_fname = os.path.join(OUTPUT_DIR, f"{stem}_output{ext}")
trail_path   = os.path.join(OUTPUT_DIR, f"{stem}_trail.jpg")

# set param
filter_kwargs = {
    # remove noise (smoothing filter, size has to be 2n-1)
    "gaussian_kernel_size": 19,
    # remove noise and enhance information (morphology operation: erosion and dilation)
    "morphology_kernel_size": 3,
    "background_threshold": 40,  # difference threshold of background /not background
}

hsv_kwargs = {
    "white_lower": [0, 0, 160],    # HSV lower bound for white (any hue, low sat, high val)
    "white_upper": [180, 50, 255], # HSV upper bound for white
    "min_radius": 5,               # smallest expected ball radius (px)
    "max_radius": 25,              # largest expected ball radius (px)
    "min_circularity": 0.4,        # 0.0~1.0, 1.0 = perfect circle
    "person_area_min": 5000,       # blobs larger than this (px²) are treated as person
}

# max pixel distance allowed between consecutive trail points (filters impossible jumps)
MAX_TRAIL_JUMP = 300
# max consecutive frames to rely on Kalman prediction without a real detection
MAX_PREDICT_FRAMES = 8

# =========================================================================


def track_baseball(video_path):
    cap = cv2.VideoCapture(video_path)

    # obtain video info ------------------------------------------------
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f'Video resolution: {width} x {height}, frame rate: {fps} fps, frame count: {frame_count}.')

    # initialize background subtractor ---------------------------------
    backSub = cv2.createBackgroundSubtractorMOG2(
        varThreshold=filter_kwargs['background_threshold'],
        detectShadows=False)

    # initialize Kalman filter (state: x, y, vx, vy  |  measurement: x, y) ----
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov      = np.eye(4, dtype=np.float32) * 1e-2
    kalman.measurementNoiseCov  = np.eye(2, dtype=np.float32) * 1e-1
    kalman.errorCovPost         = np.eye(4, dtype=np.float32)
    kalman_initialized = False
    predict_count = 0

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
    warmup_frames = 6  # skip first 0.2s for background determination
    for frame_id, frame in enumerate(tqdm(frames)):

        # apply background subtraction as mask
        fg_mask = backSub.apply(cv2.GaussianBlur(
            frame, (filter_kwargs["gaussian_kernel_size"],)*2, 0))
        result_frame = frame.copy()

        # apply morphology
        kernel = np.ones(
            (filter_kwargs["morphology_kernel_size"],)*2, np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # skip warmup frames
        if frame_id < warmup_frames:
            continue

        # person exclusion: large blobs in fg_mask → exclusion zone
        person_excl = np.zeros_like(fg_mask)
        all_cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in all_cnts:
            if cv2.contourArea(cnt) > hsv_kwargs["person_area_min"]:
                cv2.drawContours(person_excl, [cnt], -1, 255, -1)
        person_excl = cv2.dilate(person_excl, np.ones((15, 15), np.uint8))
        fg_clean = cv2.bitwise_and(fg_mask, cv2.bitwise_not(person_excl))

        # HSV white color filter
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv_frame,
            np.array(hsv_kwargs["white_lower"]),
            np.array(hsv_kwargs["white_upper"]))

        # intersect cleaned fg mask with white color mask
        ball_mask = cv2.bitwise_and(fg_clean, white_mask)

        # Kalman predict (every frame once initialized)
        predicted = None
        if kalman_initialized:
            predicted = kalman.predict()

        # find ball candidate: filter by size and circularity
        area_min = np.pi * hsv_kwargs["min_radius"] ** 2 * 0.5
        area_max = np.pi * hsv_kwargs["max_radius"] ** 2 * 3.0
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, best_circularity = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min or area > area_max:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity >= hsv_kwargs["min_circularity"] and circularity > best_circularity:
                best_circularity = circularity
                best_cnt = cnt

        # update Kalman and trail
        if best_cnt is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(best_cnt)
            center = (int(cx), int(cy))
            if not kalman_initialized:
                kalman.statePre  = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
                kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], np.float32)
                kalman_initialized = True
            kalman.correct(np.array([[cx], [cy]], np.float32))
            predict_count = 0
            if not trail or np.hypot(cx - trail[-1][0], cy - trail[-1][1]) < MAX_TRAIL_JUMP:
                trail.append(center)
            cv2.circle(result_frame, center, int(radius), (0, 0, 255), 2)   # red = detected
        elif kalman_initialized and predict_count < MAX_PREDICT_FRAMES:
            px, py = int(predicted[0]), int(predicted[1])
            predict_count += 1
            pred_center = (px, py)
            if not trail or np.hypot(px - trail[-1][0], py - trail[-1][1]) < MAX_TRAIL_JUMP:
                trail.append(pred_center)
            cv2.circle(result_frame, pred_center, hsv_kwargs["min_radius"], (0, 165, 255), 2)  # orange = predicted

        for j in range(1, len(trail)):
            cv2.line(result_frame, trail[j-1], trail[j], (255, 0, 0), 3)
        for pt in trail:
            cv2.circle(result_frame, pt, 3, (0, 255, 255), -1)  # yellow dot at each trail point

        result_frames.append(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))

    return result_frames, frames, trail, width, height, fps


result_frames, raw_frames, trail, width, height, fps = track_baseball(fname)

# draw complete trajectory on the first raw frame and save as static image
if trail and raw_frames:
    summary = raw_frames[0].copy()
    for j in range(1, len(trail)):
        cv2.line(summary, trail[j-1], trail[j], (255, 0, 0), 3)
    for pt in trail:
        cv2.circle(summary, pt, 3, (0, 255, 255), -1)
    cv2.imwrite(trail_path, summary)
    print(f"Trail image saved to {os.path.abspath(trail_path)}")


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

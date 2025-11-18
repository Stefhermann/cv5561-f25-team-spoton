from pathlib import Path

import cv2

VIDEO_PATH = "video_data/rdg_gameplay_2_5.mp4"
OUT_DIR = Path("data/frames")
NUM_FRAMES = 50

vid_cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
save_idx = 0

while True:
    ret, frame = vid_cap.read()

    if not ret:
        break

    if frame_idx % NUM_FRAMES == 0:
        out_path = OUT_DIR / f"frame_5{save_idx:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        save_idx += 1

    frame_idx += 1

vid_cap.release()
print(f"{save_idx} frames created.")

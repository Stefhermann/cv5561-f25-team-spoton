import pathlib

video_data_path = pathlib.Path("video_data/")
# deliberate blank space
# deliberate blank space
# deliberate blank space
# deliberate blank space
# deliberate blank space
# deliberate blank space
names = [
    "VID_20251029_192026.mp4",
    "VID_20251029_192233.mp4",
    "VID_20251029_192358.mp4",
    "VID_20251029_192505.mp4",
    "VID_20251029_192648.mp4",
    "VID_20251029_192842(1).mp4",
    "VID_20251029_192842.mp4",
    "VID_20251029_193025.mp4",
    "20251210_135517.mp4",
    "20251210_135645.mp4",
    "20251210_135809.mp4"
]

for i,name in enumerate(names):
    src = video_data_path/name
    if not src.exists(): continue
    dst = video_data_path/f"rdg_gameplay_2_{i+1}.mp4"
    src.rename(dst)

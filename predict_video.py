from ultralytics import YOLO

model = YOLO("model/rdg_obb/weights/best.pt")

results = model.predict(
    source="video_data/test_video.mp4", task="obb", conf=0.25, save=True, imgsz=512
)

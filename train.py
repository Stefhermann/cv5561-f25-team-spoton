from ultralytics import YOLO


def main():
    data_yaml = "data.yaml"
    model = YOLO("yolov8n-obb.pt")

    model.train(
        data=data_yaml,
        task="obb",
        epochs=50,
        imgsz=512,
        batch=16,
        project="model",
        name="rdg_obb",
        # augmentations (docs: https://docs.ultralytics.com/guides/yolo-data-augmentation/)
        hsv_h=1.0,
        hsv_v=0.2,
        degrees=180,
        scale=0.5,
        shear=5.0,
    )


if __name__ == "__main__":
    main()

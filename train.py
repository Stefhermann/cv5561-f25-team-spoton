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
        name="rdg_oob",
    )


if __name__ == "__main__":
    main()

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="data/dice/die_yolo",
        task="classify",
        epochs=50,
        imgsz=224,
        batch=16,
        project="classifier_models",
        name="die_number_classifier",
    )


if __name__ == "__main__":
    main()

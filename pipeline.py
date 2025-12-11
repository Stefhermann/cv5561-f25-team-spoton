import cv2
import numpy as np
from ultralytics import YOLO

from classify_dice import RecognitionModel
from scoring import Scoring
from spatial_association import associate_die_card
from tracking import DiceTracker
from visualization import draw_associations


def main(vid_src):
    # model = YOLO("model/rdg_obb6/weights/best.pt")
    model = RecognitionModel()

    unsupervised_imgs = np.stack([cv2.imread("data/misc/all.jpg")], axis=0)
    player_imgs = [
        np.stack([cv2.imread("data/misc/red.jpg")], axis=0),
        np.stack([cv2.imread("data/misc/yellow.jpg")], axis=0),
        np.stack([cv2.imread("data/misc/purple.jpg")], axis=0),
    ]
    model.train_player_vocab(unsupervised_imgs, player_imgs)
    print("Vocab learned")

    tracker = DiceTracker()
    scorer = Scoring()

    print("Opening video source...")
    cap = cv2.VideoCapture(vid_src)

    if not cap.isOpened():
        print("Video Source not detected")
        return

    frame_num = -1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or frame read failed.")
            break
        frame_num += 1
        if frame_num < 340:
            continue

        print(f"==== Frame {frame_num} ====")

        # results = model(frame, task = "obb", conf = 0.25, imgsz = 512)
        results = model(frame)
        res = results

        detections = []
        print(res.obb._original_res)
        for i, _ in enumerate(res.obb):
            x1, y1, x2, y2 = res.obb.xyxy[i, ...]
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "cls": int(res.obb.cls[i]),
                    "comf": float(res.obb.conf[i].item()),
                }
            )

        associations = associate_die_card(detections, res.names)

        dice_inputs = []
        for det in detections:
            cls_name = res.names[det["cls"]]
            if "_" in cls_name:
                dice_inputs.append({"class": cls_name, "bbox": det["bbox"]})

        tracked_dice = tracker.update(dice_inputs)

        scores = scorer.update_scores(tracked_dice, associations)

        frame_out = draw_associations(
            frame.copy(), associations, detections, res.names, tracked_dice, scores
        )

        cv2.imshow("🎲♠️ Rhubarb Dice Game Real Time Scoring", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main("./test_data/test_video_1.mp4")
    main("video_data/rdg_gameplay_2_9.mp4")

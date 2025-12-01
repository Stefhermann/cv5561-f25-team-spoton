import cv2
from ultralytics import YOLO

from spatial_association import associate_die_card
from tracking import DiceTracker
from scoring import Scoring
from visualization import draw_associations

def main(vid_src):
    model = YOLO("model/rdg_obb/weights/best.pt")

    tracker = DiceTracker()
    scorer = Scoring()

    cap = cv2.VideoCapture(vid_src)

    if not cap.isOpened():
        print("Video Source not detected")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or frame read failed.")
            break

        results = model(frame, task = "obb", conf = 0.25, imgsz = 512)
        res = results[0]

        detections = []
        for b in res.obb:
            x1, y1, x2, y2 = b.xyxy[0]
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cls": int(b.cls.item()),
                "comf": float(b.conf.item())
            })

        associations = associate_die_card(detections, res.names)

        dice_inputs = []
        for det in detections:
            cls_name = res.names[det["cls"]]
            if "_" in cls_name:
                dice_inputs.append({
                    "class": cls_name,
                    "bbox": det["bbox"]
                })
        
        tracked_dice = tracker.update(dice_inputs)

        scores = scorer.update_scores(tracked_dice, associations)

        frame_out = draw_associations(frame.copy(), associations, detections, res.names, tracked_dice, scores)

        cv2.imshow("🎲♠️ Rhubarb Dice Game Real Time Scoring", frame_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("./test_data/test_video_1.mp4")
    
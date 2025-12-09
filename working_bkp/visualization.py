import cv2
from spatial_association import CARD_CLASSES

def draw_associations(frame, associations, detections, names, tracked_dice = None, scores = None):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])

        cls_name = names[det["cls"]]

        color = (0, 255, 0) if cls_name in CARD_CLASSES else (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls_name, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for a in associations:
        cx, cy = map(int, a["die_centroid"])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if a["card_class"] is not None:
            cv2.putText(frame, f"->{a['card_class']}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if tracked_dice is not None:
        for die_id, (bbox, cls_name) in tracked_dice.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.putText(frame, f"ID: {die_id}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if scores is not None:
        cv2.putText(frame, f"Blue: {scores['blue']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, f"Red: {scores['red']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"Yellow: {scores['yellow']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    return frame
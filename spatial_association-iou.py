import numpy as np

CARD_CLASSES = {"ace", "two", "three", "four"}


def centroid(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / float(areaA + areaB - inter)


def extract_card_dice(detections, names):
    cards, dice = [], []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["cls"]
        cls_name = names[cls_id]

        if cls_name in CARD_CLASSES:
            cards.append({
                "class": cls_name,
                "bbox": (x1, y1, x2, y2)
            })

        else:
            dice.append({
                "class": cls_name,
                "bbox": (x1, y1, x2, y2),
                "centroid": centroid(x1, y1, x2, y2)
            })

    return cards, dice


def associate_die_card(detections, names):
    cards, dice = extract_card_dice(detections, names)

    associations = []

    for die in dice:
        die_box = list(die["bbox"])
        best_iou = 0
        assigned_card = None

        for card in cards:
            card_box = list(card["bbox"])
            val = iou(die_box, card_box)
            if val > best_iou:
                best_iou = val
                assigned_card = card["class"]

        # Require IoU threshold to avoid random matches
        if best_iou < 0.05:
            assigned_card = None

        associations.append({
            "die_class": die["class"],
            "die_centroid": die["centroid"],
            "card_class": assigned_card
        })

    return associations

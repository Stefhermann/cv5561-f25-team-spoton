import numpy as np
from card_Kalman import KalmanCardBox  # your Kalman class
CARD_FILTERS = {}

CARD_CLASSES = {"ace", "two", "three", "four"}


def centroid(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def get_filtered_bbox(card_id, bbox):
    if card_id not in CARD_FILTERS:
        CARD_FILTERS[card_id] = KalmanCardBox()
    return CARD_FILTERS[card_id].update(bbox)


def extract_card_dice(detections, names):
    cards, dice = [], []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["cls"]
        cls_name = names[cls_id]

        if cls_name in CARD_CLASSES:
            # Apply Kalman filtering for each card instance
            card_id = f"{cls_name}_{i}"
            filtered_box = get_filtered_bbox(card_id, [x1, y1, x2, y2])

            cards.append({
                "class": cls_name,
                "bbox": filtered_box
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
        dx, dy = die["centroid"]
        assigned_card = None

        for card in cards:
            x1, y1, x2, y2 = card["bbox"]
            if x1 <= dx <= x2 and y1 <= dy <= y2:
                assigned_card = card["class"]
                break

        associations.append({
            "die_class": die["class"],
            "die_centroid": die["centroid"],
            "card_class": assigned_card
        })

    return associations

import numpy as np

CARD_CLASSES = {"ace", "two", "three", "four"}


def centroid(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def extract_card_dice(detections, names):
    """
    Extracts cards and dice from a list of detections. Takes in a list of detections and a list of class names as parameters and returns two lists, one each for cards and dice
    """
    cards, dice = [], []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]  #Using bbox instead of xyxy
        cls_id = det["cls"]
        cls_name = names[cls_id]

        if cls_name in CARD_CLASSES:
            cards.append({"class": cls_name, "bbox": (x1, y1, x2, y2)})
        else:
            dice.append(
                {
                    "class": cls_name,
                    "bbox": (x1, y1, x2, y2),
                    "centroid": centroid(x1, y1, x2, y2),
                }
            )

    return cards, dice


def associate_die_card(detections, names):
    #Associates dice with cards based on if their bounding box centroids fall within the card's bounding box.
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

        associations.append(
            {
                "die_class": die["class"],
                "die_centroid": die["centroid"],
                "card_class": assigned_card,
            }
        )

    return associations

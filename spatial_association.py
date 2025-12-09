# import numpy as np
# from card_Kalman import KalmanCardBox

# CARD_FILTERS = {}
# CARD_CLASSES = {"ace", "two", "three", "four"}


# def centroid(box):
#     x1, y1, x2, y2 = box
#     return ((x1 + x2) / 2, (y1 + y2) / 2)


# def point_in_box(pt, box):
#     x, y = pt
#     x1, y1, x2, y2 = box
#     return x1 <= x <= x2 and y1 <= y <= y2


# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter <= 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return inter / (areaA + areaB - inter)


# def get_filtered_bbox(card_id, bbox):
#     if card_id not in CARD_FILTERS:
#         CARD_FILTERS[card_id] = KalmanCardBox()
#     return CARD_FILTERS[card_id].update(bbox)


# def extract_card_dice(detections, names):
#     cards = []
#     dice = []

#     for i, det in enumerate(detections):
#         x1, y1, x2, y2 = det["bbox"]
#         cls = names[det["cls"]]

#         if cls in CARD_CLASSES:
#             filtered = get_filtered_bbox(f"{cls}_{i}", [x1, y1, x2, y2])
#             cards.append({"class": cls, "bbox": filtered})

#         else:
#             dice.append({
#                 "class": cls,
#                 "bbox": [x1, y1, x2, y2],
#                 "centroid": centroid([x1, y1, x2, y2])
#             })

#     return cards, dice


# def associate_die_card(detections, names):
#     cards, dice = extract_card_dice(detections, names)
#     associations = []

#     for die in dice:
#         cx, cy = die["centroid"]
#         die_box = die["bbox"]

#         assigned_card = None

#         # 1️⃣ Try centroid-in-smoothed-card first
#         for card in cards:
#             if point_in_box((cx, cy), card["bbox"]):
#                 assigned_card = card["class"]
#                 break

#         # 2️⃣ If centroid fails, try IoU
#         if assigned_card is None:
#             best_i = 0
#             best_c = None
#             for card in cards:
#                 score = iou(die_box, card["bbox"])
#                 if score > best_i:
#                     best_i = score
#                     best_c = card["class"]
#             if best_i > 0.02:  # tiny dice → threshold must be small
#                 assigned_card = best_c

#         associations.append({
#             "die_class": die["class"],
#             "die_centroid": die["centroid"],
#             "card_class": assigned_card
#         })

#     return associations

# import numpy as np
# import cv2

# CARD_CLASSES = {"ace", "two", "three", "four"}

# def centroid(x1, y1, x2, y2):
#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0
#     return (cx, cy)

# def extract_card_dice(detections, names):
#     cards, dice = [], []

#     for det in detections:
#         x1, y1, x2, y2 = det["xyxy"][0]
#         cls_id = det["cls"]
#         cls_name = names[cls_id]

#         if cls_name in CARD_CLASSES:
#             cards.append({
#                 "class": cls_name,
#                 "bbox": (x1, y1, x2, y2)
#             })
#         else:
#             dice.append({
#                 "class": cls_name,
#                 "bbox": (x1, y1, x2, y2),
#                 "centroid": centroid(x1, y1, x2, y2)
#             })

#     return cards, dice

# def associate_die_card(detection, names):
#     cards, dice = extract_card_dice(detection, names)

#     associations = []

#     for die in dice:
#         dx, dy = die["centroid"]
#         assigned_card = None
#         for card in cards:
#             x1, y1, x2, y2 = card["bbox"]
#             if x1 <= dx <= x2 and y1 <= dy <= y2:
#                 assigned_card = card["class"]
#                 break
        
#         associations.append({
#             "die_class": die["class"],
#             "die_centroid": die["centroid"],
#             "card_class": assigned_card
#         })
#     return associations

# spatial_association.py

# import numpy as np
# import cv2

# CARD_CLASSES = {"ace", "two", "three", "four"}

# # ------------ Kalman for cards ------------

# class KalmanCardBox:
#     def __init__(self):
#         # state: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
#         # meas:  [x1, y1, x2, y2]
#         self.kf = cv2.KalmanFilter(8, 4)

#         self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
#         self.kf.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)

#         self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
#         for i in range(4):
#             self.kf.transitionMatrix[i, i + 4] = 1.0

#         self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
#         self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 3e-1

#         self.initialized = False

#     def update(self, bbox):
#         # bbox: [x1, y1, x2, y2]
#         m = np.array(bbox, dtype=np.float32).reshape(4, 1)

#         if not self.initialized:
#             self.kf.statePre[:4] = m
#             self.kf.statePre[4:] = 0
#             self.initialized = True

#         self.kf.predict()
#         corrected = self.kf.correct(m)
#         return corrected[:4].reshape(4).tolist()


# _CARD_FILTERS = {}  # key: string id → KalmanCardBox


# def _get_filtered_bbox(card_id, bbox):
#     if card_id not in _CARD_FILTERS:
#         _CARD_FILTERS[card_id] = KalmanCardBox()
#     return _CARD_FILTERS[card_id].update(bbox)


# # ------------ API used by the pipeline ------------

# def extract_cards(detections, names):
#     """
#     detections: list of { 'bbox': [x1,y1,x2,y2], 'cls': int, ... }
#     names: id → class_name

#     returns: list of { 'class': str, 'bbox': [x1,y1,x2,y2] } with Kalman smoothing
#     """
#     cards = []
#     for i, det in enumerate(detections):
#         x1, y1, x2, y2 = det["bbox"]
#         cls_name = names[det["cls"]]

#         if cls_name in CARD_CLASSES:
#             filtered = _get_filtered_bbox(f"{cls_name}_{i}", [x1, y1, x2, y2])
#             cards.append({
#                 "class": cls_name,
#                 "bbox": filtered
#             })

#     return cards


# GEM
# spatial_association.py
# spatial_association.py
# import numpy as np

# CARD_CLASSES = {"ace", "two", "three", "four"}

# def centroid(x1, y1, x2, y2):
#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0
#     return (cx, cy)


# def extract_card_dice(detections, names):
#     cards, dice = [], []

#     for det in detections:
#         x1, y1, x2, y2 = det["bbox"]
#         cls_id = det["cls"]
#         cls_name = names[cls_id]

#         if cls_name in CARD_CLASSES:
#             cards.append({
#                 "class": cls_name,
#                 "bbox": (x1, y1, x2, y2)
#             })
#         else:
#             dice.append({
#                 "class": cls_name,
#                 "bbox": (x1, y1, x2, y2),
#                 "centroid": centroid(x1, y1, x2, y2)
#             })

#     return cards, dice


# def associate_die_card(detections, names):
#     cards, dice = extract_card_dice(detections, names)

#     associations = []

#     # Padding factor: 0.1 means we shrink the card box by 10% on each side.
#     # The die centroid must be in this "safe zone" to count.
#     MARGIN_FACTOR = 0.1 

#     for die in dice:
#         dx, dy = die["centroid"]
#         assigned_card = None

#         for card in cards:
#             x1, y1, x2, y2 = card["bbox"]
#             w = x2 - x1
#             h = y2 - y1
            
#             # Calculate Safe Zone (Inner Box)
#             sx1 = x1 + (w * MARGIN_FACTOR)
#             sy1 = y1 + (h * MARGIN_FACTOR)
#             sx2 = x2 - (w * MARGIN_FACTOR)
#             sy2 = y2 - (h * MARGIN_FACTOR)

#             # Check if die centroid is inside the Safe Zone
#             if sx1 <= dx <= sx2 and sy1 <= dy <= sy2:
#                 assigned_card = card["class"]
#                 break # Die can only be on one card

#         associations.append({
#             "die_class": die["class"],
#             "die_centroid": die["centroid"],
#             "card_class": assigned_card
#         })

#     return associations

# Cl
# spatial_association.py - Improved Die-Card Association

import numpy as np

CARD_CLASSES = {"ace", "two", "three", "four"}

def centroid(x1, y1, x2, y2):
    """Calculate centroid of bounding box."""
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class CardSmoother:
    """
    Simple exponential moving average for card bounding boxes.
    Reduces jitter without Kalman complexity.
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.boxes = {}  # card_key -> smoothed_bbox
    
    def update(self, card_key, bbox):
        """Apply EMA smoothing to card bbox."""
        if card_key not in self.boxes:
            self.boxes[card_key] = list(bbox)
            return self.boxes[card_key]
        
        old = self.boxes[card_key]
        smoothed = [
            self.alpha * old[i] + (1 - self.alpha) * bbox[i]
            for i in range(4)
        ]
        self.boxes[card_key] = smoothed
        return smoothed


# Global smoother instance
_CARD_SMOOTHER = CardSmoother(alpha=0.7)


def extract_card_dice(detections, names):
    """
    Separate detections into cards and dice.
    Apply smoothing to card boxes to reduce jitter.
    
    Args:
        detections: list of {"bbox": [x1,y1,x2,y2], "cls": int, ...}
        names: dict mapping class_id to class_name
        
    Returns:
        cards: list of {"class": str, "bbox": [x1,y1,x2,y2]}
        dice: list of {"class": str, "bbox": tuple, "centroid": tuple}
    """
    cards, dice = [], []

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["cls"]
        cls_name = names[cls_id]

        if cls_name in CARD_CLASSES:
            # Apply smoothing to reduce card bbox jitter
            card_key = f"{cls_name}_{i}"
            smoothed = _CARD_SMOOTHER.update(card_key, [x1, y1, x2, y2])
            
            cards.append({
                "class": cls_name,
                "bbox": smoothed
            })
        else:
            # Dice use raw detection (tracker handles smoothing)
            dice.append({
                "class": cls_name,
                "bbox": (x1, y1, x2, y2),
                "centroid": centroid(x1, y1, x2, y2)
            })

    return cards, dice


def associate_die_card(detections, names):
    """
    Associate dice with cards using conservative margin-based containment.
    
    Strategy:
    - Shrink card boxes by margin to create "safe zone"
    - Only score if die centroid is in safe zone
    - Prevents edge-case false positives
    
    Args:
        detections: list of detection dicts
        names: class name mapping
        
    Returns:
        list of associations: [{"die_class", "die_centroid", "card_class"}, ...]
    """
    cards, dice = extract_card_dice(detections, names)
    associations = []

    # Conservative margin - die must be clearly inside card
    # Reduced to catch more edge cases while still preventing false positives
    MARGIN_FACTOR = 0.08  # 8% shrink on each side (was 0.15)

    for die in dice:
        dx, dy = die["centroid"]
        assigned_card = None

        # Check each card to see if die is in its safe zone
        for card in cards:
            x1, y1, x2, y2 = card["bbox"]
            w = x2 - x1
            h = y2 - y1
            
            # Calculate safe zone (shrunken box)
            sx1 = x1 + (w * MARGIN_FACTOR)
            sy1 = y1 + (h * MARGIN_FACTOR)
            sx2 = x2 - (w * MARGIN_FACTOR)
            sy2 = y2 - (h * MARGIN_FACTOR)

            # Check containment in safe zone
            if sx1 <= dx <= sx2 and sy1 <= dy <= sy2:
                assigned_card = card["class"]
                break  # Die can only be on one card

        associations.append({
            "die_class": die["class"],
            "die_centroid": die["centroid"],
            "card_class": assigned_card
        })

    return associations
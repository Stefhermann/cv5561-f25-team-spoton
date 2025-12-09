# import cv2
# from spatial_association import CARD_CLASSES

# def draw_associations(frame, associations, detections, names, tracked_dice = None, scores = None):
#     for det in detections:
#         x1, y1, x2, y2 = map(int, det["bbox"])

#         cls_name = names[det["cls"]]

#         color = (0, 255, 0) if cls_name in CARD_CLASSES else (255, 0, 0)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, cls_name, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     for a in associations:
#         cx, cy = map(int, a["die_centroid"])
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#         if a["card_class"] is not None:
#             cv2.putText(frame, f"->{a['card_class']}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#     if tracked_dice is not None:
#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             x1, y1, x2, y2 = map(int, bbox)
#             cv2.putText(frame, f"ID: {die_id}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#     if scores is not None:
#         cv2.putText(frame, f"Blue: {scores['blue']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#         cv2.putText(frame, f"Red: {scores['red']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#         cv2.putText(frame, f"Yellow: {scores['yellow']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

#     return frame

# visualization.py - Enhanced with debugging info

import cv2
from spatial_association import CARD_CLASSES

def draw_associations(frame, associations, detections, names, tracked_dice=None, scores=None):
    """
    Enhanced visualization with debugging information.
    Shows card safe zones and detailed die state.
    """
    
    # Draw all detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = names[det["cls"]]
        
        # Different colors for cards vs dice
        if cls_name in CARD_CLASSES:
            color = (0, 255, 0)  # Green for cards
        else:
            color = (255, 0, 0)  # Blue for dice
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls_name, (x1, y1-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw die centroids and associations
    for a in associations:
        cx, cy = map(int, a["die_centroid"])
        
        # Color code based on association status
        if a["card_class"] is not None:
            centroid_color = (0, 255, 255)  # Yellow - on card
        else:
            centroid_color = (128, 128, 128)  # Gray - not on card
        
        cv2.circle(frame, (cx, cy), 5, centroid_color, -1)
        
        # Show association arrow
        if a["card_class"] is not None:
            cv2.putText(frame, f"->{a['card_class']}", 
                       (cx + 10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw tracked dice IDs with state info
    if tracked_dice is not None:
        for die_id, (bbox, cls_name) in tracked_dice.items():
            x1, y1, x2, y2 = map(int, bbox)
            
            # Show ID prominently
            cv2.putText(frame, f"ID: {die_id}", 
                       (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw scores with breakdown
    if scores is not None:
        y_offset = 40
        for color_name, score in scores.items():
            if color_name == "blue":
                text_color = (255, 0, 0)
            elif color_name == "red":
                text_color = (0, 0, 255)
            else:  # yellow
                text_color = (0, 255, 255)
            
            text = f"{color_name.capitalize()}: {score}"
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
            y_offset += 40
    
    return frame


def draw_associations_debug(frame, associations, detections, names, tracked_dice=None, 
                            scores=None, scorer=None):
    """
    ENHANCED DEBUG VERSION - Shows card margins and die states.
    Use this version to diagnose scoring issues.
    """
    
    # First draw basic associations
    frame = draw_associations(frame, associations, detections, names, tracked_dice, scores)
    
    # Draw card safe zones (margins)
    MARGIN_FACTOR = 0.08  # Must match spatial_association.py
    
    for det in detections:
        cls_name = names[det["cls"]]
        if cls_name in CARD_CLASSES:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            
            # Calculate and draw safe zone
            sx1 = int(x1 + (w * MARGIN_FACTOR))
            sy1 = int(y1 + (h * MARGIN_FACTOR))
            sx2 = int(x2 - (w * MARGIN_FACTOR))
            sy2 = int(y2 - (h * MARGIN_FACTOR))
            
            # Draw inner "safe zone" rectangle in yellow
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
    
    # Show detailed die state information
    if scorer is not None and hasattr(scorer, 'die_states'):
        y_pos = 200
        cv2.putText(frame, "DIE STATES:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 25
        
        for die_id, state in scorer.die_states.items():
            info = (f"ID{die_id}: {state['color']}_{state['value']} | "
                   f"Pending:{state.get('pending_card', '?')} | "
                   f"Confirmed:{state.get('confirmed_card', '?')} | "
                   f"Frames:{state.get('stable_frames', 0)}")
            
            cv2.putText(frame, info, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
    
    return frame
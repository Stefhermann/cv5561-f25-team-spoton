# import numpy as np
# import math

# def center(bbox):
#     x1, y1, x2, y2 = bbox
#     return ( (x1+x2)/2, (y1+y2)/2 )


# def squared_dist(p1, p2):
#     return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


# class DiceTracker:
#     """
#     More stable tracker:
#       - Matches dice by centroid distance
#       - Avoids creating new IDs due to small bbox changes
#       - Dice must be far apart to get new IDs
#     """
#     def __init__(self, max_distance=80, max_missing=10):
#         self.next_id = 0
#         self.tracks = {}      # id -> (bbox, cls_name, miss_count)
#         self.max_distance = max_distance
#         self.max_missing = max_missing

#     def update(self, detections):
#         if len(self.tracks) == 0:
#             # First frame → assign initial IDs
#             for det in detections:
#                self.tracks[self.next_id] = [ det["bbox"], det["class"], 0 ]
#                self.next_id += 1
#             return {tid: (t[0], t[1]) for tid, t in self.tracks.items()} 
       
#         #compute centroids
#         track_ids = list(self.tracks.keys())
#         track_centers = [center(self.tracks[tid][0]) for tid in track_ids]
#         det_centers = [center(det["bbox"]) for det in detections]

#         used_dets = set()
#         new_tracks = {}

#         for tid, t_center in zip(track_ids, track_centers):
#             best_det = None
#             best_dist = float("inf")

#             for i, d_center in enumerate(det_centers):
#                 if i in used_dets:
#                     continue
#                 dist = squared_dist(t_center, d_center)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_det = i

#             if best_det is not None and math.sqrt(best_dist) < self.max_distance:
#                 # => matched
#                 new_tracks[tid] = [detections[best_det]["bbox"], detections[best_det]["class"], 0]
#                 used_dets.add(best_det)
#             else:
#                 # no match => increase miss count
#                 old_bbox, old_class, miss = self.tracks[tid]
#                 miss += 1
#                 if miss < self.max_missing:
#                     new_tracks[tid] = [old_bbox, old_class, miss]
        
#         for i, det in enumerate(detections):
#             if i not in used_dets:
#                 new_tracks[self.next_id] = [det["bbox"], det["class"], 0]
#                 self.next_id += 1

#         self.tracks = new_tracks

#         return {tid : (t[0], t[1]) for tid, t in self.tracks.items()}

# import numpy as np
# import math

# def center(bbox):
#     x1, y1, x2, y2 = bbox
#     return ( (x1+x2)/2, (y1+y2)/2 )


# def squared_dist(p1, p2):
#     return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


# class DiceTracker:
#     """
#     More stable tracker:
#       - Matches dice by centroid distance AND class name
#       - Increased tolerance for distance and missing frames
#     """
#     # Increased max_distance and max_missing for a stickier, more robust track.
#     def __init__(self, max_distance=120, max_missing=20): # Increased from 80 and 10
#         self.next_id = 0
#         self.tracks = {}      # id -> (bbox, cls_name, miss_count)
#         self.max_distance = max_distance
#         self.max_missing = max_missing

#     def update(self, detections):
#         if len(self.tracks) == 0:
#             # First frame → assign initial IDs
#             for det in detections:
#                self.tracks[self.next_id] = [ det["bbox"], det["class"], 0 ]
#                self.next_id += 1
#             return {tid: (t[0], t[1]) for tid, t in self.tracks.items()} 
       
#         #compute centroids
#         track_ids = list(self.tracks.keys())
#         track_centers = [center(self.tracks[tid][0]) for tid in track_ids]
#         det_centers = [center(det["bbox"]) for det in detections]

#         used_dets = set()
#         new_tracks = {}

#         for tid, t_center in zip(track_ids, track_centers):
#             best_det = None
#             best_dist = float("inf")
            
#             # Get the expected class of the tracked object (e.g., 'red_6')
#             expected_class = self.tracks[tid][1]

#             for i, d_center in enumerate(det_centers):
#                 if i in used_dets:
#                     continue
                
#                 # CRITICAL FIX: Only consider a detection as a match if it has the same class
#                 if detections[i]["class"] != expected_class:
#                     continue
                
#                 dist = squared_dist(t_center, d_center)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_det = i

#             if best_det is not None and math.sqrt(best_dist) < self.max_distance:
#                 # => matched
#                 new_tracks[tid] = [detections[best_det]["bbox"], detections[best_det]["class"], 0]
#                 used_dets.add(best_det)
#             else:
#                 # no match => increase miss count
#                 old_bbox, old_class, miss = self.tracks[tid]
#                 miss += 1
#                 if miss < self.max_missing:
#                     new_tracks[tid] = [old_bbox, old_class, miss]
        
#         for i, det in enumerate(detections):
#             if i not in used_dets:
#                 new_tracks[self.next_id] = [det["bbox"], det["class"], 0]
#                 self.next_id += 1

#         self.tracks = new_tracks

#         return {tid : (t[0], t[1]) for tid, t in self.tracks.items()}

# tracking.py - Improved Stable Dice Tracking

# import numpy as np
# import math

# def center(bbox):
#     """Calculate center point of bounding box."""
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)


# def squared_dist(p1, p2):
#     """Calculate squared Euclidean distance between two points."""
#     return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


# class DiceTracker:
#     """
#     Stable dice tracker that:
#     - Matches dice by class AND proximity
#     - Uses strict distance thresholds to prevent ID swapping
#     - Maintains tracks through temporary occlusions
#     - Smooths bounding boxes to reduce jitter
#     """
    
#     def __init__(self, max_distance=100, max_missing=25, bbox_smoothing=0.7):
#         """
#         Args:
#             max_distance: Max pixel distance for matching (stricter = more stable)
#             max_missing: Frames to keep track alive without detection
#             bbox_smoothing: EMA smoothing factor (0=no smoothing, 1=full smoothing)
#         """
#         self.next_id = 0
#         self.tracks = {}  # id -> (bbox, cls_name, miss_count, ema_bbox)
#         self.max_distance = max_distance
#         self.max_missing = max_missing
#         self.bbox_smoothing = bbox_smoothing

#     def _smooth_bbox(self, new_bbox, old_bbox):
#         """Apply exponential moving average smoothing to bounding box."""
#         if old_bbox is None:
#             return new_bbox
        
#         alpha = self.bbox_smoothing
#         smoothed = [
#             alpha * old + (1 - alpha) * new 
#             for old, new in zip(old_bbox, new_bbox)
#         ]
#         return smoothed

#     def _check_duplicate_detections(self, detections):
#         """
#         Check for and remove duplicate detections of the same die.
#         This prevents one die from getting multiple IDs.
#         """
#         if len(detections) <= 1:
#             return detections
        
#         filtered = []
#         for i, det1 in enumerate(detections):
#             is_duplicate = False
#             for j, det2 in enumerate(filtered):
#                 # Same class and very close positions = duplicate
#                 if det1["class"] == det2["class"]:
#                     c1 = center(det1["bbox"])
#                     c2 = center(det2["bbox"])
#                     dist = math.sqrt(squared_dist(c1, c2))
                    
#                     # If two detections of same class are within 30 pixels, keep only one
#                     if dist < 30:
#                         is_duplicate = True
#                         break
            
#             if not is_duplicate:
#                 filtered.append(det1)
        
#         if len(filtered) < len(detections):
#             print(f"⚠️  Removed {len(detections) - len(filtered)} duplicate detections")
        
#         return filtered

#     def update(self, detections):
#         """
#         Update tracks with new detections.
        
#         Args:
#             detections: list of {"class": str, "bbox": [x1,y1,x2,y2]}
            
#         Returns:
#             dict: {track_id: (bbox, cls_name)}
#         """
#         # Remove duplicate detections first
#         detections = self._check_duplicate_detections(detections)
        
#         # First frame - initialize all tracks
#         if len(self.tracks) == 0:
#             for det in detections:
#                 self.tracks[self.next_id] = {
#                     "bbox": det["bbox"],
#                     "class": det["class"],
#                     "miss_count": 0,
#                     "ema_bbox": det["bbox"][:]
#                 }
#                 self.next_id += 1
#             return {tid: (t["bbox"], t["class"]) for tid, t in self.tracks.items()}

#         # Prepare data for matching
#         track_ids = list(self.tracks.keys())
#         track_centers = [center(self.tracks[tid]["ema_bbox"]) for tid in track_ids]
#         det_centers = [center(det["bbox"]) for det in detections]

#         used_dets = set()
#         new_tracks = {}

#         # MATCHING PHASE - Assign detections to existing tracks
#         for tid, t_center in zip(track_ids, track_centers):
#             best_det_idx = None
#             best_dist = float("inf")
            
#             # Get the expected class for this track
#             expected_class = self.tracks[tid]["class"]

#             # Find closest detection with matching class
#             for det_idx, d_center in enumerate(det_centers):
#                 if det_idx in used_dets:
#                     continue
                
#                 # CRITICAL: Only match detections with same class
#                 if detections[det_idx]["class"] != expected_class:
#                     continue
                
#                 dist = squared_dist(t_center, d_center)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_det_idx = det_idx

#             # If match found within distance threshold, update track
#             if best_det_idx is not None and math.sqrt(best_dist) < self.max_distance:
#                 det = detections[best_det_idx]
                
#                 # Apply bbox smoothing to reduce jitter
#                 old_ema = self.tracks[tid]["ema_bbox"]
#                 new_ema = self._smooth_bbox(det["bbox"], old_ema)
                
#                 new_tracks[tid] = {
#                     "bbox": det["bbox"],  # Keep raw bbox
#                     "class": det["class"],
#                     "miss_count": 0,
#                     "ema_bbox": new_ema  # Smoothed bbox for matching
#                 }
#                 used_dets.add(best_det_idx)
#             else:
#                 # No match - increment miss count
#                 old_track = self.tracks[tid]
#                 miss = old_track["miss_count"] + 1
                
#                 # Keep track alive if under miss threshold
#                 if miss < self.max_missing:
#                     new_tracks[tid] = {
#                         "bbox": old_track["bbox"],
#                         "class": old_track["class"],
#                         "miss_count": miss,
#                         "ema_bbox": old_track["ema_bbox"]
#                     }

#         # NEW TRACKS PHASE - Create tracks for unmatched detections
#         for det_idx, det in enumerate(detections):
#             if det_idx not in used_dets:
#                 new_tracks[self.next_id] = {
#                     "bbox": det["bbox"],
#                     "class": det["class"],
#                     "miss_count": 0,
#                     "ema_bbox": det["bbox"][:]
#                 }
#                 self.next_id += 1

#         self.tracks = new_tracks

#         # Return simplified format for downstream use
#         return {tid: (t["bbox"], t["class"]) for tid, t in self.tracks.items()}

# tracking.py - Improved Stable Dice Tracking

import numpy as np
import math

def center(bbox):
    """Calculate center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def squared_dist(p1, p2):
    """Calculate squared Euclidean distance between two points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


class DiceTracker:
    """
    Stable dice tracker that:
    - Matches dice by class AND proximity
    - Uses strict distance thresholds to prevent ID swapping
    - Maintains tracks through temporary occlusions
    - Smooths bounding boxes to reduce jitter
    """
    
    def __init__(self, max_distance=100, max_missing=25, bbox_smoothing=0.7):
        """
        Args:
            max_distance: Max pixel distance for matching (stricter = more stable)
            max_missing: Frames to keep track alive without detection
            bbox_smoothing: EMA smoothing factor (0=no smoothing, 1=full smoothing)
        """
        self.next_id = 0
        self.tracks = {}  # id -> (bbox, cls_name, miss_count, ema_bbox)
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.bbox_smoothing = bbox_smoothing

    def _smooth_bbox(self, new_bbox, old_bbox):
        """Apply exponential moving average smoothing to bounding box."""
        if old_bbox is None:
            return new_bbox
        
        alpha = self.bbox_smoothing
        smoothed = [
            alpha * old + (1 - alpha) * new 
            for old, new in zip(old_bbox, new_bbox)
        ]
        return smoothed

    def _check_duplicate_detections(self, detections):
        """
        Check for and remove duplicate detections of the same die.
        This prevents one die from getting multiple IDs.
        Uses AGGRESSIVE filtering with IoU + distance checks.
        """
        if len(detections) <= 1:
            return detections
        
        def compute_iou(bbox1, bbox2):
            """Calculate IoU between two bounding boxes."""
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        filtered = []
        for i, det1 in enumerate(detections):
            is_duplicate = False
            
            for j, det2 in enumerate(filtered):
                # Only check same class
                if det1["class"] != det2["class"]:
                    continue
                
                # Check 1: Distance-based (increased threshold to 80 pixels)
                c1 = center(det1["bbox"])
                c2 = center(det2["bbox"])
                dist = math.sqrt(squared_dist(c1, c2))
                
                # Check 2: IoU-based (any overlap at all is suspicious for dice)
                iou = compute_iou(det1["bbox"], det2["bbox"])
                
                # If same class AND (close distance OR any overlap), it's a duplicate
                if dist < 80 or iou > 0.1:
                    is_duplicate = True
                    print(f"  🗑️  Filtered duplicate: {det1['class']} (dist={dist:.1f}px, IoU={iou:.3f})")
                    break
            
            if not is_duplicate:
                filtered.append(det1)
        
        if len(filtered) < len(detections):
            print(f"⚠️  Removed {len(detections) - len(filtered)} duplicate detections in this frame")
        
        return filtered

    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: list of {"class": str, "bbox": [x1,y1,x2,y2]}
            
        Returns:
            dict: {track_id: (bbox, cls_name)}
        """
        # Remove duplicate detections first
        detections = self._check_duplicate_detections(detections)
        
        # First frame - initialize all tracks
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "miss_count": 0,
                    "ema_bbox": det["bbox"][:]
                }
                self.next_id += 1
            return {tid: (t["bbox"], t["class"]) for tid, t in self.tracks.items()}

        # Prepare data for matching
        track_ids = list(self.tracks.keys())
        track_centers = [center(self.tracks[tid]["ema_bbox"]) for tid in track_ids]
        det_centers = [center(det["bbox"]) for det in detections]

        used_dets = set()
        new_tracks = {}

        # MATCHING PHASE - Assign detections to existing tracks
        for tid, t_center in zip(track_ids, track_centers):
            best_det_idx = None
            best_dist = float("inf")
            
            # Get the expected class for this track
            expected_class = self.tracks[tid]["class"]

            # Find closest detection with matching class
            for det_idx, d_center in enumerate(det_centers):
                if det_idx in used_dets:
                    continue
                
                # CRITICAL: Only match detections with same class
                if detections[det_idx]["class"] != expected_class:
                    continue
                
                dist = squared_dist(t_center, d_center)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            # If match found within distance threshold, update track
            if best_det_idx is not None and math.sqrt(best_dist) < self.max_distance:
                det = detections[best_det_idx]
                
                # Apply bbox smoothing to reduce jitter
                old_ema = self.tracks[tid]["ema_bbox"]
                new_ema = self._smooth_bbox(det["bbox"], old_ema)
                
                new_tracks[tid] = {
                    "bbox": det["bbox"],  # Keep raw bbox
                    "class": det["class"],
                    "miss_count": 0,
                    "ema_bbox": new_ema  # Smoothed bbox for matching
                }
                used_dets.add(best_det_idx)
            else:
                # No match - increment miss count
                old_track = self.tracks[tid]
                miss = old_track["miss_count"] + 1
                
                # Keep track alive if under miss threshold
                if miss < self.max_missing:
                    new_tracks[tid] = {
                        "bbox": old_track["bbox"],
                        "class": old_track["class"],
                        "miss_count": miss,
                        "ema_bbox": old_track["ema_bbox"]
                    }

        # NEW TRACKS PHASE - Create tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                new_tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "class": det["class"],
                    "miss_count": 0,
                    "ema_bbox": det["bbox"][:]
                }
                self.next_id += 1

        self.tracks = new_tracks

        # Return simplified format for downstream use
        return {tid: (t["bbox"], t["class"]) for tid, t in self.tracks.items()}
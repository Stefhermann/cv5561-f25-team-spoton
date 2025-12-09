import numpy as np
import math

def center(bbox):
    x1, y1, x2, y2 = bbox
    return ( (x1+x2)/2, (y1+y2)/2 )


def squared_dist(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


class DiceTracker:
    """
    More stable tracker:
      - Matches dice by centroid distance
      - Avoids creating new IDs due to small bbox changes
      - Dice must be far apart to get new IDs
    """
    def __init__(self, max_distance=80, max_missing=10):
        self.next_id = 0
        self.tracks = {}      # id -> (bbox, cls_name, miss_count)
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, detections):
        if len(self.tracks) == 0:
            # First frame → assign initial IDs
            for det in detections:
               self.tracks[self.next_id] = [ det["bbox"], det["class"], 0 ]
               self.next_id += 1
            return {tid: (t[0], t[1]) for tid, t in self.tracks.items()} 
       
        #compute centroids
        track_ids = list(self.tracks.keys())
        track_centers = [center(self.tracks[tid][0]) for tid in track_ids]
        det_centers = [center(det["bbox"]) for det in detections]

        used_dets = set()
        new_tracks = {}

        for tid, t_center in zip(track_ids, track_centers):
            best_det = None
            best_dist = float("inf")

            for i, d_center in enumerate(det_centers):
                if i in used_dets:
                    continue
                dist = squared_dist(t_center, d_center)
                if dist < best_dist:
                    best_dist = dist
                    best_det = i

            if best_det is not None and math.sqrt(best_dist) < self.max_distance:
                # => matched
                new_tracks[tid] = [detections[best_det]["bbox"], detections[best_det]["class"], 0]
                used_dets.add(best_det)
            else:
                # no match => increase miss count
                old_bbox, old_class, miss = self.tracks[tid]
                miss += 1
                if miss < self.max_missing:
                    new_tracks[tid] = [old_bbox, old_class, miss]
        
        for i, det in enumerate(detections):
            if i not in used_dets:
                new_tracks[self.next_id] = [det["bbox"], det["class"], 0]
                self.next_id += 1

        self.tracks = new_tracks

        return {tid : (t[0], t[1]) for tid, t in self.tracks.items()}
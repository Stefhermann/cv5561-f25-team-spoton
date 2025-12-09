# analyze_detections.py - Understand why YOLO creates duplicates

import cv2
import numpy as np
from ultralytics import YOLO

def analyze_duplicate_pattern(vid_src, target_frame=706):
    """
    Analyze detection patterns at a specific frame to understand
    why YOLO creates duplicate detections.
    """
    model = YOLO("model/rdg_obb/weights/best.pt")
    cap = cv2.VideoCapture(vid_src)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count != target_frame:
            continue
        
        print(f"\n{'='*80}")
        print(f"ANALYZING FRAME {frame_count}")
        print(f"{'='*80}\n")
        
        # Get detections with different settings
        print("Testing different YOLO configurations:\n")
        
        configs = [
            {"conf": 0.25, "iou": 0.45, "name": "Default"},
            {"conf": 0.30, "iou": 0.30, "name": "Aggressive NMS"},
            {"conf": 0.35, "iou": 0.25, "name": "Very Aggressive"},
            {"conf": 0.40, "iou": 0.20, "name": "Extremely Aggressive"},
        ]
        
        for config in configs:
            results = model(
                frame,
                task="obb",
                conf=config["conf"],
                iou=config["iou"],
                imgsz=512,
                agnostic_nms=False
            )
            r = results[0]
            
            # Count by class
            class_counts = {}
            for b in r.obb:
                cls_name = r.names[int(b.cls.item())]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            # Focus on blue dice
            blue_dets = []
            for b in r.obb:
                cls_name = r.names[int(b.cls.item())]
                if cls_name.startswith("blue"):
                    x1, y1, x2, y2 = b.xyxy[0]
                    blue_dets.append({
                        "class": cls_name,
                        "conf": float(b.conf.item()),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            print(f"{config['name']} (conf={config['conf']}, iou={config['iou']}):")
            print(f"  Total detections: {len(r.obb)}")
            print(f"  Blue dice: {sum(1 for c in class_counts if c.startswith('blue'))}")
            
            if blue_dets:
                print(f"  Blue detections breakdown:")
                blue_class_counts = {}
                for det in blue_dets:
                    cls = det["class"]
                    blue_class_counts[cls] = blue_class_counts.get(cls, 0) + 1
                
                for cls, count in sorted(blue_class_counts.items()):
                    if count > 1:
                        print(f"    ⚠️  {cls}: {count} detections (DUPLICATE!)")
                        
                        # Show distances between duplicates
                        cls_dets = [d for d in blue_dets if d["class"] == cls]
                        for i in range(len(cls_dets)):
                            for j in range(i+1, len(cls_dets)):
                                d1, d2 = cls_dets[i], cls_dets[j]
                                
                                # Calculate centroid distance
                                c1x = (d1["bbox"][0] + d1["bbox"][2]) / 2
                                c1y = (d1["bbox"][1] + d1["bbox"][3]) / 2
                                c2x = (d2["bbox"][0] + d2["bbox"][2]) / 2
                                c2y = (d2["bbox"][1] + d2["bbox"][3]) / 2
                                dist = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)
                                
                                # Calculate IoU
                                x1 = max(d1["bbox"][0], d2["bbox"][0])
                                y1 = max(d1["bbox"][1], d2["bbox"][1])
                                x2 = min(d1["bbox"][2], d2["bbox"][2])
                                y2 = min(d1["bbox"][3], d2["bbox"][3])
                                
                                intersection = max(0, x2-x1) * max(0, y2-y1)
                                area1 = (d1["bbox"][2]-d1["bbox"][0]) * (d1["bbox"][3]-d1["bbox"][1])
                                area2 = (d2["bbox"][2]-d2["bbox"][0]) * (d2["bbox"][3]-d2["bbox"][1])
                                iou = intersection / (area1 + area2 - intersection) if (area1+area2-intersection) > 0 else 0
                                
                                print(f"      Distance: {dist:.1f}px, IoU: {iou:.3f}, "
                                      f"Conf: {d1['conf']:.3f} vs {d2['conf']:.3f}")
                    else:
                        print(f"    ✅ {cls}: 1 detection")
            print()
        
        break
    
    cap.release()
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("\nIf duplicates have:")
    print("  • Distance > 100px AND IoU < 0.1:")
    print("    → YOLO is seeing dice twice due to rotation/lighting")
    print("    → Use AGGRESSIVE deduplication (1 per class)")
    print("\n  • Distance < 50px AND IoU > 0.3:")
    print("    → Normal NMS failure")
    print("    → Lower iou threshold (try 0.2)")
    print("\n  • Similar confidence scores:")
    print("    → Model is genuinely confused")
    print("    → May need model retraining")
    print()

if __name__ == "__main__":
    # Change target_frame to the frame number from your screenshot
    analyze_duplicate_pattern("./test_data/test_video_8.mp4", target_frame=1966)
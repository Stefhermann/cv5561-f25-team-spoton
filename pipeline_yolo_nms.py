# import cv2
# import numpy as np
# from ultralytics import YOLO
# from spatial_association import associate_die_card
# from visualization import draw_associations
# from tracking import DiceTracker
# from scoring import Scoring

# def apply_class_wise_nms(detections, names, iou_threshold=0.3):
#     """
#     Apply Non-Maximum Suppression separately for each class.
#     This prevents duplicate detections of the same die.
    
#     Args:
#         detections: list of detection dicts
#         names: class names mapping
#         iou_threshold: IoU threshold for NMS (lower = more aggressive)
    
#     Returns:
#         filtered detections
#     """
#     if len(detections) == 0:
#         return detections
    
#     # Group by class
#     class_groups = {}
#     for det in detections:
#         cls_name = names[det["cls"]]
#         if cls_name not in class_groups:
#             class_groups[cls_name] = []
#         class_groups[cls_name].append(det)
    
#     filtered = []
#     removed_count = 0
    
#     # Apply NMS to each class separately
#     for cls_name, dets in class_groups.items():
#         if len(dets) == 1:
#             filtered.extend(dets)
#             continue
        
#         # Convert to numpy arrays for NMS
#         boxes = np.array([d["bbox"] for d in dets])
#         scores = np.array([d["conf"] for d in dets])
        
#         # OpenCV NMS
#         indices = cv2.dnn.NMSBoxes(
#             bboxes=boxes.tolist(),
#             scores=scores.tolist(),
#             score_threshold=0.0,  # Already filtered by YOLO
#             nms_threshold=iou_threshold
#         )
        
#         # Keep only selected detections
#         if len(indices) > 0:
#             indices = indices.flatten()
#             for idx in indices:
#                 filtered.append(dets[idx])
            
#             removed = len(dets) - len(indices)
#             if removed > 0:
#                 removed_count += removed
#                 print(f"  🗑️  NMS removed {removed} duplicate {cls_name} detection(s)")
    
#     if removed_count > 0:
#         print(f"⚠️  Total duplicates removed by NMS: {removed_count}")
    
#     return filtered


# def main(vid_src, use_nms=True, nms_threshold=0.3):
#     """
#     Main pipeline with optional NMS filtering.
    
#     Args:
#         vid_src: video source
#         use_nms: whether to apply NMS filtering
#         nms_threshold: IoU threshold (0.1-0.5, lower=more aggressive)
#     """
#     model = YOLO("model/rdg_obb/weights/best.pt")
    
#     tracker = DiceTracker()
#     scorer = Scoring()

#     cap = cv2.VideoCapture(vid_src)

#     if not cap.isOpened():
#         print("Video/Camera not detected")
#         return
    
#     print(f"\n{'='*60}")
#     print(f"STARTING PIPELINE")
#     print(f"NMS Filtering: {'ENABLED' if use_nms else 'DISABLED'}")
#     if use_nms:
#         print(f"NMS Threshold: {nms_threshold} (lower = more aggressive)")
#     print(f"{'='*60}\n")
    
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("End of video or frame read failed.")
#             break

#         frame_count += 1
        
#         results = model(frame, task="obb", conf=0.50, imgsz=512)
#         r = results[0]

#         detections = []
#         for b in r.obb:
#             x1, y1, x2, y2 = b.xyxy[0]
#             detections.append({
#                 "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                 "cls": int(b.cls.item()),
#                 "conf": float(b.conf.item())
#             })
        
#         # Apply NMS filtering to remove duplicates
#         if use_nms and len(detections) > 0:
#             detections = apply_class_wise_nms(detections, r.names, nms_threshold)
    
#         # Spatial association
#         associations = associate_die_card(detections, r.names)

#         dice_inputs = []
#         for det in detections:
#             cls_name = r.names[det["cls"]]
#             if "_" in cls_name:  # dice class
#                 dice_inputs.append({
#                     "class": cls_name,
#                     "bbox": det["bbox"]
#                 })

#         # Tracking
#         tracked_dice = tracker.update(dice_inputs)

#         # Scoring
#         scores = scorer.update_scores(tracked_dice, associations)

#         # Visualization
#         output_frame = draw_associations(
#             frame.copy(), associations, detections, r.names, 
#             tracked_dice, scores
#         )

#         # Show stats on frame
#         cv2.putText(output_frame, f"Frame: {frame_count}", 
#                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(output_frame, f"Detections: {len(detections)}", 
#                    (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(output_frame, f"Tracks: {len(tracked_dice)}", 
#                    (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         cv2.imshow("Rhubarb Real-Time Scoring", output_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Final summary
#     print(f"\n{'='*60}")
#     print("FINAL SUMMARY")
#     print(f"{'='*60}")
#     print(f"Total frames: {frame_count}")
#     print(f"Final scores: {scores}")
#     print(f"\nFinal die states ({len(scorer.die_states)} total):")
    
#     # Group by color
#     for color in ["blue", "red", "yellow"]:
#         color_dice = [(did, st) for did, st in scorer.die_states.items() 
#                      if st["color"] == color]
#         if color_dice:
#             print(f"\n{color.upper()}:")
#             for die_id, state in color_dice:
#                 card = state.get("confirmed_card")
#                 if card:
#                     from scoring import CARD_VALUES
#                     points = state["value"] * CARD_VALUES.get(card, 0)
#                     print(f"  ID {die_id}: {color}_{state['value']} on {card} = {points} points")
#                 else:
#                     print(f"  ID {die_id}: {color}_{state['value']} (not on card)")

# if __name__ == "__main__":
#     # Run with NMS enabled (RECOMMENDED)
#     main("./test_data/test_video_8.mp4", use_nms=True, nms_threshold=0.5)
    
#     # If NMS is too aggressive and removes real dice, increase threshold:
#     # main("./test_data/test_video_3.mp4", use_nms=True, nms_threshold=0.5)
    
#     # Or disable NMS (not recommended for your case):
#     # main("./test_data/test_video_3.mp4", use_nms=False)

# pipeline_final.py - Complete solution with all fixes

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from spatial_association import associate_die_card
# from visualization import draw_associations
# from tracking import DiceTracker
# from scoring import Scoring

# def deduplicate_by_class(detections, names):
#     """
#     NUCLEAR OPTION: Keep only ONE detection per class.
    
#     Use this if:
#     - You have unique dice (no two dice have same color+value)
#     - Standard NMS doesn't work
#     - You're getting 2-3x duplicates consistently
#     """
#     class_best = {}
    
#     for det in detections:
#         cls_name = names[det["cls"]]
        
#         if cls_name not in class_best:
#             class_best[cls_name] = det
#         else:
#             # Keep detection with higher confidence
#             if det["conf"] > class_best[cls_name]["conf"]:
#                 class_best[cls_name] = det
    
#     removed = len(detections) - len(class_best)
#     if removed > 0:
#         print(f"  🗑️  Class deduplication removed {removed} duplicates")
    
#     return list(class_best.values())


# def main(vid_src, enable_class_dedup=True):
#     """
#     Final pipeline with all fixes combined.
    
#     Args:
#         enable_class_dedup: If True, keeps only 1 detection per class (RECOMMENDED)
#     """
#     model = YOLO("model/rdg_obb/weights/best.pt")
    
#     # More conservative tracker settings
#     tracker = DiceTracker(
#         max_distance=120,    # Increased - be more lenient about matching
#         max_missing=30,      # Increased - keep tracks alive longer
#         bbox_smoothing=0.75  # More smoothing
#     )
    
#     scorer = Scoring()

#     cap = cv2.VideoCapture(vid_src)

#     if not cap.isOpened():
#         print("Video/Camera not detected")
#         return
    
#     print(f"\n{'='*70}")
#     print("FINAL COMPLETE SOLUTION")
#     print(f"Class Deduplication: {'ENABLED' if enable_class_dedup else 'DISABLED'}")
#     print(f"{'='*70}\n")
    
#     frame_count = 0
#     prev_scores = {"red": 0, "blue": 0, "yellow": 0}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("End of video or frame read failed.")
#             break

#         frame_count += 1
        
#         # Stage 1: YOLO with aggressive settings
#         results = model(
#             frame, 
#             task="obb",
#             conf=0.35,           # Higher confidence
#             iou=0.25,            # Aggressive NMS
#             imgsz=512,
#             max_det=50,
#             agnostic_nms=False   # Class-aware
#         )
#         r = results[0]

#         # Stage 2: Extract detections
#         detections = []
#         for b in r.obb:
#             x1, y1, x2, y2 = b.xyxy[0]
#             detections.append({
#                 "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                 "cls": int(b.cls.item()),
#                 "conf": float(b.conf.item())
#             })
        
#         original_count = len(detections)
        
#         # Stage 3: Additional class-based deduplication
#         if enable_class_dedup:
#             detections = deduplicate_by_class(detections, r.names)
        
#         # Stage 4: Spatial association
#         associations = associate_die_card(detections, r.names)

#         # Stage 5: Extract dice for tracking
#         dice_inputs = []
#         for det in detections:
#             cls_name = r.names[det["cls"]]
#             if "_" in cls_name:
#                 dice_inputs.append({
#                     "class": cls_name,
#                     "bbox": det["bbox"]
#                 })

#         # Stage 6: Tracking
#         tracked_dice = tracker.update(dice_inputs)

#         # Stage 7: Scoring
#         scores = scorer.update_scores(tracked_dice, associations)
        
#         # Monitor for score changes
#         for color in ["blue", "red", "yellow"]:
#             if scores[color] != prev_scores[color]:
#                 print(f"[Frame {frame_count}] {color.upper()}: {prev_scores[color]} → {scores[color]}")
#         prev_scores = scores.copy()

#         # Stage 8: Visualization
#         output_frame = draw_associations(
#             frame.copy(), associations, detections, r.names, 
#             tracked_dice, scores
#         )

#         # Show detailed stats
#         cv2.putText(output_frame, f"Frame: {frame_count}", 
#                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(output_frame, f"Detections: {len(detections)}", 
#                    (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(output_frame, f"Tracks: {len(tracked_dice)}", 
#                    (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         if enable_class_dedup and original_count != len(detections):
#             cv2.putText(output_frame, f"Filtered: {original_count - len(detections)}", 
#                        (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         cv2.imshow("Rhubarb Real-Time Scoring", output_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Final summary
#     print(f"\n{'='*70}")
#     print("FINAL SUMMARY")
#     print(f"{'='*70}")
#     print(f"Total frames: {frame_count}")
#     print(f"Final scores: {scores}")
    
#     # Detailed breakdown
#     from scoring import CARD_VALUES
#     print(f"\nDice on cards:")
#     for color in ["blue", "red", "yellow"]:
#         color_dice = [(did, st) for did, st in scorer.die_states.items() 
#                      if st["color"] == color and st.get("confirmed_card") in CARD_VALUES]
        
#         if color_dice:
#             print(f"\n{color.upper()}: {scores[color]} points")
#             for die_id, state in color_dice:
#                 card = state["confirmed_card"]
#                 points = state["value"] * CARD_VALUES[card]
#                 print(f"  ID {die_id}: {color}_{state['value']} on {card} → {state['value']}×{CARD_VALUES[card]} = {points}")
#         else:
#             print(f"\n{color.upper()}: {scores[color]} points (no dice on cards)")
    
#     # Check for potential issues
#     print(f"\n{'='*70}")
#     print("VERIFICATION:")
#     print(f"{'='*70}")
    
#     all_dice_on_cards = [
#         (st["color"], st["value"], st["confirmed_card"])
#         for st in scorer.die_states.values()
#         if st.get("confirmed_card") in CARD_VALUES
#     ]
    
#     # Check for duplicate dice
#     seen = set()
#     duplicates = []
#     for color, value, card in all_dice_on_cards:
#         key = (color, value)
#         if key in seen:
#             duplicates.append(f"{color}_{value}")
#         seen.add(key)
    
#     if duplicates:
#         print(f"⚠️  WARNING: Duplicate dice detected: {duplicates}")
#         print("   This means the same die has multiple IDs!")
#         print("   The deduplication may need to be more aggressive.")
#     else:
#         print("✅ No duplicate dice detected - all unique!")
    
#     # Calculate expected vs actual
#     expected_blue = sum(
#         st["value"] * CARD_VALUES[st["confirmed_card"]]
#         for st in scorer.die_states.values()
#         if st["color"] == "blue" and st.get("confirmed_card") in CARD_VALUES
#     )
    
#     if expected_blue != scores["blue"]:
#         print(f"\n⚠️  MISMATCH: Expected blue={expected_blue}, got {scores['blue']}")
#     else:
#         print(f"\n✅ Score calculation verified: blue={scores['blue']}")

# if __name__ == "__main__":
#     # Run with class deduplication enabled (RECOMMENDED)
#     main("./test_data/test_video_8.mp4", enable_class_dedup=False)
    
    # If you have multiple dice with same color+value, disable:
    # main("./test_data/test_video_3.mp4", enable_class_dedup=False)
# import cv2
# from ultralytics import YOLO
# from spatial_association import associate_die_card
# from visualization import draw_associations
# from tracking import DiceTracker
# from scoring import Scoring

# def main(vid_src):
#     model = YOLO("model/rdg_obb/weights/best.pt")
        
#     tracker = DiceTracker()
#     scorer = Scoring()

#     cap = cv2.VideoCapture(vid_src)

#     if not cap.isOpened():
#         print("Video/Camera not detected")
#         return
    
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("End of video or frame read failed.")
#             break

#         results = model(frame, task="obb", conf=0.25, imgsz=512)
#         r = results[0]

#         detections = []
#         for b in r.obb:
#             x1, y1, x2, y2 = b.xyxy[0]  # flatten
#             detections.append({
#                 "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                 "cls": int(b.cls.item()),
#                 "conf": float(b.conf.item())
#             })
    
#         # Spatial association
#         associations = associate_die_card(detections, r.names)

#         dice_inputs = []
#         for det in detections:
#             cls_name = r.names[det["cls"]]
#             if "_" in cls_name:     # dice class
#                 dice_inputs.append({
#                     "class": cls_name,
#                     "bbox": det["bbox"]
#                 })

#         # Tracking
#         tracked_dice = tracker.update(dice_inputs)

#         # Scoring
#         scores = scorer.update_scores(tracked_dice, associations)

#         # Visualization
#         output_frame = draw_associations(frame.copy(), associations, detections, r.names, tracked_dice, scores)

#         cv2.imshow("Rhubarb Real-Time Scoring", output_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main("./test_data/test_video_8.mp4")
#     # main(0)


# import cv2
# from ultralytics import YOLO
# from spatial_association import associate_die_card
# from visualization import draw_associations, draw_associations_debug
# from tracking import DiceTracker
# from scoring import Scoring

# def main(vid_src, debug_mode=True):
#     """
#     Main pipeline with debug mode option.
    
#     Args:
#         vid_src: Video source (file path or 0 for webcam)
#         debug_mode: If True, shows detailed debug info and prints scoring events
#     """
#     model = YOLO("model/rdg_obb/weights/best.pt")
    
#     tracker = DiceTracker()
#     scorer = Scoring()

#     cap = cv2.VideoCapture(vid_src)

#     if not cap.isOpened():
#         print("Video/Camera not detected")
#         return
    
#     frame_count = 0
#     prev_scores = {"red": 0, "blue": 0, "yellow": 0}
    
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("End of video or frame read failed.")
#             break

#         frame_count += 1
        
#         results = model(frame, task="obb", conf=0.25, imgsz=512)
#         r = results[0]

#         detections = []
#         for b in r.obb:
#             x1, y1, x2, y2 = b.xyxy[0]
#             detections.append({
#                 "bbox": [float(x1), float(y1), float(x2), float(y2)],
#                 "cls": int(b.cls.item()),
#                 "conf": float(b.conf.item())
#             })
    
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

#         # DEBUG: Print score changes
#         if debug_mode:
#             for color in ["red", "blue", "yellow"]:
#                 if scores[color] != prev_scores[color]:
#                     print(f"[Frame {frame_count}] {color.upper()} score changed: "
#                           f"{prev_scores[color]} -> {scores[color]}")
                    
#                     # Show which dice are contributing
#                     contributing_dice = []
#                     for die_id, state in scorer.die_states.items():
#                         if state["color"] == color and state.get("confirmed_card") is not None:
#                             card = state["confirmed_card"]
#                             from scoring import CARD_VALUES
#                             points = state["value"] * CARD_VALUES.get(card, 0)
#                             contributing_dice.append(
#                                 f"ID{die_id}({state['value']}×{card}={points})"
#                             )
                    
#                     if contributing_dice:
#                         print(f"  Contributing: {', '.join(contributing_dice)}")
            
#             prev_scores = scores.copy()

#         # Visualization
#         if debug_mode:
#             output_frame = draw_associations_debug(
#                 frame.copy(), associations, detections, r.names, 
#                 tracked_dice, scores, scorer
#             )
#         else:
#             output_frame = draw_associations(
#                 frame.copy(), associations, detections, r.names, 
#                 tracked_dice, scores
#             )

#         cv2.imshow("Rhubarb Real-Time Scoring", output_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Print final summary
#     if debug_mode:
#         print("\n=== FINAL SUMMARY ===")
#         print(f"Total frames: {frame_count}")
#         print(f"Final scores: {scores}")
#         print(f"Active dice: {len(scorer.die_states)}")
#         print("\nDie states:")
#         for die_id, state in scorer.die_states.items():
#             print(f"  ID {die_id}: {state['color']}_{state['value']} -> "
#                   f"confirmed={state.get('confirmed_card', 'None')}")

# if __name__ == "__main__":
#     # Use debug mode to diagnose issues
#     main("./test_data/test_video_8.mp4", debug_mode=True)
#     # main(0, debug_mode=True)  # For webcam

import cv2
from ultralytics import YOLO
from spatial_association import associate_die_card
from visualization import draw_associations
from tracking import DiceTracker
from scoring import Scoring, CARD_VALUES

def main(vid_src, debug_mode=True):
    """
    Enhanced debug pipeline to diagnose scoring issues.
    """
    model = YOLO("model/rdg_obb/weights/best.pt")
    
    tracker = DiceTracker()
    scorer = Scoring()

    cap = cv2.VideoCapture(vid_src)

    if not cap.isOpened():
        print("Video/Camera not detected")
        return
    
    frame_count = 0
    prev_scores = {"red": 0, "blue": 0, "yellow": 0}
    
    print("\n" + "="*80)
    print("STARTING ENHANCED DEBUG MODE")
    print("="*80 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or frame read failed.")
            break

        frame_count += 1
        
        results = model(frame, task="obb", conf=0.25, imgsz=512)
        r = results[0]

        detections = []
        for b in r.obb:
            x1, y1, x2, y2 = b.xyxy[0]
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cls": int(b.cls.item()),
                "conf": float(b.conf.item())
            })
    
        # Spatial association
        associations = associate_die_card(detections, r.names)

        dice_inputs = []
        for det in detections:
            cls_name = r.names[det["cls"]]
            if "_" in cls_name:  # dice class
                dice_inputs.append({
                    "class": cls_name,
                    "bbox": det["bbox"]
                })

        # Tracking
        tracked_dice = tracker.update(dice_inputs)

        # Scoring
        scores = scorer.update_scores(tracked_dice, associations)

        # ============ ENHANCED DEBUG OUTPUT ============
        if debug_mode:
            # Check for score changes
            for color in ["red", "blue", "yellow"]:
                if scores[color] != prev_scores[color]:
                    print(f"\n{'='*80}")
                    print(f"[Frame {frame_count}] ⚠️  {color.upper()} SCORE CHANGED: {prev_scores[color]} → {scores[color]}")
                    print(f"{'='*80}")
                    
                    # Show ALL dice of this color
                    print(f"\n📊 ALL {color.upper()} DICE IN FRAME:")
                    blue_dice_in_frame = [d for d in dice_inputs if d["class"].startswith(color)]
                    if blue_dice_in_frame:
                        for i, die in enumerate(blue_dice_in_frame, 1):
                            print(f"  {i}. Detection: {die['class']} at {die['bbox']}")
                    else:
                        print(f"  ⚠️  NO {color.upper()} DICE DETECTED IN THIS FRAME!")
                    
                    # Show all tracked dice of this color
                    print(f"\n🎯 ALL TRACKED {color.upper()} DICE:")
                    tracked_blue = {tid: (bbox, cls) for tid, (bbox, cls) in tracked_dice.items() 
                                   if cls.startswith(color)}
                    if tracked_blue:
                        for tid, (bbox, cls) in tracked_blue.items():
                            print(f"  ID {tid}: {cls} at {bbox}")
                    else:
                        print(f"  ⚠️  NO TRACKED {color.upper()} DICE!")
                    
                    # Show associations for this color
                    print(f"\n🔗 ASSOCIATIONS FOR {color.upper()} DICE:")
                    blue_assoc = [a for a in associations if a["die_class"].startswith(color)]
                    if blue_assoc:
                        for a in blue_assoc:
                            print(f"  {a['die_class']} → {a['card_class']} at {a['die_centroid']}")
                    else:
                        print(f"  ⚠️  NO ASSOCIATIONS FOR {color.upper()} DICE!")
                    
                    # Show scorer state for this color
                    print(f"\n🎲 SCORER STATE FOR {color.upper()} DICE:")
                    contributing_dice = []
                    for die_id, state in scorer.die_states.items():
                        if state["color"] == color:
                            card = state.get("confirmed_card")
                            pending = state.get("pending_card")
                            frames = state.get("stable_frames", 0)
                            missing = state.get("missing_frames", 0)
                            
                            if card in CARD_VALUES:
                                points = state["value"] * CARD_VALUES[card]
                                contributing_dice.append(die_id)
                                status = "✅ SCORING"
                            else:
                                points = 0
                                status = "❌ NOT SCORING"
                            
                            print(f"  ID {die_id}: {state['color']}_{state['value']}")
                            print(f"    Status: {status}")
                            print(f"    Pending: {pending} | Confirmed: {card} | Frames: {frames} | Missing: {missing}")
                            print(f"    Points: {points}")
                    
                    # Show calculation breakdown
                    print(f"\n🧮 SCORE CALCULATION FOR {color.upper()}:")
                    total = 0
                    for die_id, state in scorer.die_states.items():
                        if state["color"] == color:
                            card = state.get("confirmed_card")
                            if card in CARD_VALUES:
                                points = state["value"] * CARD_VALUES[card]
                                total += points
                                print(f"  ID {die_id}: {state['value']} × {CARD_VALUES[card]} ({card}) = {points}")
                    print(f"  {'─'*40}")
                    print(f"  TOTAL: {total}")
                    
                    if total != scores[color]:
                        print(f"  ⚠️⚠️⚠️ MISMATCH! Expected {total} but got {scores[color]} ⚠️⚠️⚠️")
                    
                    print(f"\n{'='*80}\n")
            
            prev_scores = scores.copy()
            
            # Every 30 frames, print a summary
            if frame_count % 30 == 0:
                print(f"\n[Frame {frame_count}] 📸 SNAPSHOT:")
                print(f"  Scores: Blue={scores['blue']}, Red={scores['red']}, Yellow={scores['yellow']}")
                print(f"  Active tracks: {len(tracker.tracks)}")
                print(f"  Die states: {len(scorer.die_states)}")

        # Visualization
        output_frame = draw_associations(frame.copy(), associations, detections, r.names, tracked_dice, scores)

        cv2.imshow("Rhubarb Real-Time Scoring", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total frames: {frame_count}")
    print(f"Final scores: {scores}")
    print(f"\nFinal die states:")
    for die_id, state in scorer.die_states.items():
        card = state.get("confirmed_card")
        if card:
            points = state["value"] * CARD_VALUES.get(card, 0)
            print(f"  ID {die_id}: {state['color']}_{state['value']} on {card} = {points} points")
        else:
            print(f"  ID {die_id}: {state['color']}_{state['value']} (not on card)")

if __name__ == "__main__":
    main("./test_data/test_video_4.mp4", debug_mode=True)
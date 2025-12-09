# diagnostic_30.py - Find out why blue is getting 30 points

"""
This script will help diagnose the "30 points instead of 12" issue.

Possible causes of 30:
1. Three dice scoring: 6+12+12=30 or 10+10+10=30
2. Two dice scoring: 18+12=30 or 24+6=30 or 10+20=30
3. Wrong die value: 15×2=30 (misdetection)
4. One die being scored multiple times due to ID instability

Run this and look for:
- Multiple blue dice with different IDs
- Same physical die getting multiple IDs
- Wrong die values being detected
"""

import cv2
from ultralytics import YOLO
from spatial_association import associate_die_card
from tracking import DiceTracker
from scoring import Scoring, CARD_VALUES

def analyze_blue_30(vid_src):
    model = YOLO("model/rdg_obb/weights/best.pt")
    tracker = DiceTracker()
    scorer = Scoring()
    cap = cv2.VideoCapture(vid_src)
    
    frame_count = 0
    found_30 = False
    
    print("\n🔍 SEARCHING FOR BLUE=30 SCENARIO...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
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
        
        associations = associate_die_card(detections, r.names)
        
        dice_inputs = []
        for det in detections:
            cls_name = r.names[det["cls"]]
            if "_" in cls_name:
                dice_inputs.append({
                    "class": cls_name,
                    "bbox": det["bbox"]
                })
        
        tracked_dice = tracker.update(dice_inputs)
        scores = scorer.update_scores(tracked_dice, associations)
        
        # Check if blue score is around 30
        if 28 <= scores["blue"] <= 32 and not found_30:
            found_30 = True
            
            print(f"{'='*80}")
            print(f"🎯 FOUND IT! Frame {frame_count}: Blue = {scores['blue']}")
            print(f"{'='*80}\n")
            
            # Analysis 1: Raw detections
            print("📷 RAW YOLO DETECTIONS IN THIS FRAME:")
            blue_dets = [det for det in detections if r.names[det["cls"]].startswith("blue")]
            print(f"  Total blue detections: {len(blue_dets)}")
            for i, det in enumerate(blue_dets, 1):
                cls_name = r.names[det["cls"]]
                bbox = det["bbox"]
                conf = det["conf"]
                print(f"  {i}. {cls_name} at {[round(x,1) for x in bbox]} (conf={conf:.2f})")
            
            # Analysis 2: Tracked dice
            print(f"\n🎯 TRACKED BLUE DICE:")
            blue_tracks = {tid: (bbox, cls) for tid, (bbox, cls) in tracked_dice.items() 
                          if cls.startswith("blue")}
            print(f"  Total blue tracks: {len(blue_tracks)}")
            for tid, (bbox, cls) in blue_tracks.items():
                print(f"  ID {tid}: {cls} at {[round(x,1) for x in bbox]}")
            
            # Analysis 3: Scorer state
            print(f"\n🎲 SCORER STATE FOR BLUE:")
            blue_states = {did: st for did, st in scorer.die_states.items() if st["color"] == "blue"}
            print(f"  Total blue die states: {len(blue_states)}")
            
            total_check = 0
            for die_id, state in blue_states.items():
                card = state.get("confirmed_card")
                pending = state.get("pending_card")
                frames = state.get("stable_frames", 0)
                
                if card in CARD_VALUES:
                    points = state["value"] * CARD_VALUES[card]
                    total_check += points
                    status = f"✅ SCORING: {state['value']}×{CARD_VALUES[card]}={points}"
                else:
                    points = 0
                    status = "❌ NOT SCORING"
                
                print(f"  ID {die_id}: blue_{state['value']}")
                print(f"    {status}")
                print(f"    Pending: {pending} | Confirmed: {card} | Frames: {frames}")
            
            print(f"\n  Total calculated: {total_check}")
            print(f"  Actual score: {scores['blue']}")
            
            if total_check != scores["blue"]:
                print(f"  ⚠️⚠️⚠️ MISMATCH DETECTED! ⚠️⚠️⚠️")
            
            # Analysis 4: Check for duplicates
            print(f"\n🔍 DUPLICATE CHECK:")
            blue_values = [state["value"] for state in blue_states.values()]
            if len(blue_values) != len(set(blue_values)):
                print(f"  ⚠️  DUPLICATE DIE VALUES FOUND: {blue_values}")
                print(f"  This suggests the same die has multiple IDs!")
            else:
                print(f"  ✅ No duplicate die values: {blue_values}")
            
            # Analysis 5: Distance between blue dice
            print(f"\n📏 DISTANCES BETWEEN BLUE DICE:")
            blue_track_list = list(blue_tracks.items())
            for i in range(len(blue_track_list)):
                for j in range(i+1, len(blue_track_list)):
                    tid1, (bbox1, cls1) = blue_track_list[i]
                    tid2, (bbox2, cls2) = blue_track_list[j]
                    
                    cx1 = (bbox1[0] + bbox1[2]) / 2
                    cy1 = (bbox1[1] + bbox1[3]) / 2
                    cx2 = (bbox2[0] + bbox2[2]) / 2
                    cy2 = (bbox2[1] + bbox2[3]) / 2
                    
                    import math
                    dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                    
                    print(f"  ID {tid1} ({cls1}) ↔ ID {tid2} ({cls2}): {dist:.1f} pixels")
                    
                    if dist < 50:
                        print(f"    ⚠️  TOO CLOSE! Likely same die with duplicate IDs!")
            
            print(f"\n{'='*80}\n")
            
            # Save the problematic frame
            cv2.imwrite(f"debug_frame_{frame_count}_blue_{scores['blue']}.jpg", frame)
            print(f"💾 Saved problematic frame as: debug_frame_{frame_count}_blue_{scores['blue']}.jpg\n")
            
            # Continue to see if score changes
            print("⏯️  Continuing to monitor...\n")
    
    cap.release()
    
    if not found_30:
        print("❌ Did not find blue score around 30 in this video.")
    else:
        print("\n✅ Analysis complete. Check the output above for the root cause.")

if __name__ == "__main__":
    analyze_blue_30("./test_data/test_video_8.mp4")
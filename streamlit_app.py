# streamlit_app.py - Real-Time Scoreboard Dashboard

import streamlit as st
import cv2
import time
import threading
from queue import Queue
from ultralytics import YOLO
from spatial_association import associate_die_card
from tracking import DiceTracker
from scoring import Scoring

# Page configuration
st.set_page_config(
    page_title="Rhubarb Dice Game - Live Scores",
    page_icon="🎲",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .score-container {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
    }
    .blue-score {
        background-color: #3498db;
        color: white;
    }
    .red-score {
        background-color: #e74c3c;
        color: white;
    }
    .yellow-score {
        background-color: #f1c40f;
        color: #333;
    }
    .status-indicator {
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .status-live {
        background-color: #2ecc71;
        color: white;
    }
    .status-stopped {
        background-color: #95a5a6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scores' not in st.session_state:
    st.session_state.scores = {"blue": 0, "red": 0, "yellow": 0}
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'score_queue' not in st.session_state:
    st.session_state.score_queue = Queue()
if 'video_thread' not in st.session_state:
    st.session_state.video_thread = None


def process_video(vid_src, score_queue, stop_event):
    """
    Background thread to process video and update scores.
    """
    model = YOLO("model/rdg_obb/weights/best.pt")
    tracker = DiceTracker()
    scorer = Scoring()
    
    cap = cv2.VideoCapture(vid_src)
    
    if not cap.isOpened():
        score_queue.put({"error": "Video/Camera not detected"})
        return
    
    frame_count = 0
    last_update_time = time.time()
    UPDATE_INTERVAL = 5.0  # Update dashboard every 5 seconds
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            score_queue.put({"status": "finished"})
            break
        
        frame_count += 1
        
        # Process frame
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
            if "_" in cls_name:
                dice_inputs.append({
                    "class": cls_name,
                    "bbox": det["bbox"]
                })
        
        # Tracking and scoring
        tracked_dice = tracker.update(dice_inputs)
        scores = scorer.update_scores(tracked_dice, associations)
        
        # Update dashboard every 5 seconds
        current_time = time.time()
        if current_time - last_update_time >= UPDATE_INTERVAL:
            score_queue.put({
                "scores": scores.copy(),
                "frame": frame_count,
                "timestamp": current_time
            })
            last_update_time = current_time
    
    cap.release()
    score_queue.put({"status": "stopped"})


def start_video_processing(vid_src):
    """Start video processing in background thread."""
    if st.session_state.is_running:
        return
    
    st.session_state.stop_event = threading.Event()
    st.session_state.video_thread = threading.Thread(
        target=process_video,
        args=(vid_src, st.session_state.score_queue, st.session_state.stop_event),
        daemon=True
    )
    st.session_state.video_thread.start()
    st.session_state.is_running = True


def stop_video_processing():
    """Stop video processing."""
    if st.session_state.is_running:
        st.session_state.stop_event.set()
        st.session_state.is_running = False


# Main UI
st.markdown('<h1 class="main-title">🎲 Rhubarb Dice Game</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Live Scoreboard</p>', unsafe_allow_html=True)

# Control buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if not st.session_state.is_running:
        if st.button("▶️ Start Game", use_container_width=True):
            start_video_processing("./test_data/test_video_4.mp4")
            st.rerun()
            
    else:
        if st.button("⏹️ Stop Game", use_container_width=True):
            stop_video_processing()
            st.rerun()

# Status indicator
if st.session_state.is_running:
    st.markdown('<div class="status-indicator status-live">🟢 LIVE - Updating every 5 seconds</div>', 
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status-indicator status-stopped">⚫ STOPPED</div>', 
                unsafe_allow_html=True)

# Check for score updates
if st.session_state.is_running:
    while not st.session_state.score_queue.empty():
        update = st.session_state.score_queue.get()
        
        if "scores" in update:
            st.session_state.scores = update["scores"]
        elif "status" in update:
            if update["status"] == "finished":
                st.session_state.is_running = False
                st.info("📹 Video playback finished!")
            elif update["status"] == "stopped":
                st.session_state.is_running = False

# Display scores
st.markdown("---")

# Blue Score
st.markdown(f'''
<div class="score-container blue-score">
    🔵 BLUE: {st.session_state.scores["blue"]}
</div>
''', unsafe_allow_html=True)

# Red Score
st.markdown(f'''
<div class="score-container red-score">
    🔴 RED: {st.session_state.scores["red"]}
</div>
''', unsafe_allow_html=True)

# Yellow Score
st.markdown(f'''
<div class="score-container yellow-score">
    🟡 YELLOW: {st.session_state.scores["yellow"]}
</div>
''', unsafe_allow_html=True)

# Auto-refresh when running
if st.session_state.is_running:
    time.sleep(0.5)  # Check for updates every 0.5 seconds
    st.rerun()
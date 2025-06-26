import streamlit as st
import cv2
import threading
import time

from blink_detector import BlinkDetector
from posture_detector import PostureDetector
from emotion_detector import EmotionDetector
from audio_analysis import start_audio_threads, get_current_audio_emotion

# App setup
st.set_page_config(page_title="Real-Time Multimodal Analyzer", layout="wide")
st.title("🎥 Real-Time Multimodal Emotion Analyzer")

FRAME_WINDOW = st.empty()
status_box = st.empty()

# Globals to communicate with thread
frame_lock = threading.Lock()
latest_frame = None
latest_stats = ""
stop_event = threading.Event()

# Detectors
blink = BlinkDetector()
posture = PostureDetector()
emotion = EmotionDetector()

def camera_loop():
    global latest_frame, latest_stats
    print("[INFO] Starting camera thread")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start_audio_threads()

    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blink.update(frame_gray)
        emo_scores = emotion.scores(frame_rgb)
        pose_label = posture.classify(frame_rgb)
        audio_emotion = get_current_audio_emotion()

        stats = f"""🧠 Emotion: {max(emo_scores, key=emo_scores.get)}
📏 Posture: {pose_label}
👁️ Blinks: {blink.total}
🎤 Audio Emotion: {audio_emotion}"""

        with frame_lock:
            latest_frame = frame_rgb.copy()
            latest_stats = stats

        time.sleep(0.05)

    cap.release()
    print("[INFO] Camera released")

# UI Buttons
col1, col2 = st.columns(2)
if "thread" not in st.session_state:
    st.session_state.thread = None

with col1:
    if st.button("▶️ Start Analysis"):
        if st.session_state.thread is None or not st.session_state.thread.is_alive():
            stop_event.clear()
            t = threading.Thread(target=camera_loop, daemon=True)
            t.start()
            st.session_state.thread = t

with col2:
    if st.button("⏹️ Stop"):
        stop_event.set()
        st.session_state.thread = None

# Main display (polling loop)
if st.session_state.get("thread", None) is not None:
    with frame_lock:
        if latest_frame is not None:
            FRAME_WINDOW.image(latest_frame, channels="RGB")
            status_box.markdown(f"```{latest_stats}```")
        else:
            FRAME_WINDOW.markdown("⌛ **Waiting for first frame...**")
else:
    status_box.info("Click 'Start Analysis' to begin real-time detection.")

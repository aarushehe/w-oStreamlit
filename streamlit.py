import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2, time, pandas as pd
import threading, csv
from collections import Counter
from datetime import datetime
from av import VideoFrame
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from blink_detector import BlinkDetector
from posture_detector import PostureDetector
from emotion_detector import EmotionDetector
from audio_analysis import start_audio_threads, get_current_audio_emotion

# Global configs and initialization
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
conf_em = ["neutral", "surprise", "angry", "happy"]
anx_em = ["fear", "sad", "disgust"]
OUTPUT_FRAME_CSV = "session_log.csv"
OUTPUT_BLINK_CSV = "blink_log.csv"
PROCESS_EVERY_N  = 5

blink   = BlinkDetector()
posture = PostureDetector()
emotion = EmotionDetector()

frame_log = {"timestamp": [], "posture": [], "audio_emotion": []}
for e in emotion.EMOS: frame_log[e] = []

start_audio_threads()
start_time = time.time()
frame_no = 0

# Verdict logic
def get_final_verdict(dominant_emotion, blink_rate, posture_state, voice_emotion):
    score = 0
    if dominant_emotion in conf_em:
        score += 1
    if 10 <= blink_rate < 25:
        score += 1
    if posture_state == "Upright":
        score += 1
    if voice_emotion == "confident":
        score += 1
    return "Confident" if score >= 3 else "Anxious"

# Save function
def save_dictlist_csv(data, filename):
    if not data: return
    with open(filename, 'w', newline='') as f:
        fieldnames = data[0].keys() if isinstance(data, list) else data.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(data, list):
            writer.writerows(data)
        else:
            rows = [dict(zip(fieldnames, vals)) for vals in zip(*data.values())]
            writer.writerows(rows)

if "summary_data" not in st.session_state:
    st.session_state.summary_data = None

def generate_summary_and_pdf(processor):
    frame_log = processor.frame_log
    if len(frame_log["timestamp"]) == 0:
        return None

    # Save CSVs
    save_dictlist_csv(frame_log, OUTPUT_FRAME_CSV)
    save_dictlist_csv(blink.log, OUTPUT_BLINK_CSV)

    df = pd.DataFrame(frame_log)
    p_counts = Counter(df["posture"])
    upright_like = p_counts.get("Upright", 0) + p_counts.get("No person", 0)
    upright_pct = (upright_like / len(df) * 100) if len(df) else 0
    dom_posture = "Upright" if upright_pct >= 20 else "Slouched"

    emo_means = df[emotion.EMOS].mean()
    dom_emotion_label = emo_means.idxmax()
    dom_emotion = "Confident" if dom_emotion_label in conf_em else "Anxious"

    if len(df["timestamp"]) >= 2:
        duration_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 60
    else:
        duration_min = 1e-6

    blink_rate = processor.blink.total / duration_min if duration_min > 0 else 0

    audio_counts = Counter(df["audio_emotion"])
    dom_audio = audio_counts.most_common(1)[0][0] if audio_counts else "Unknown"

    verdict = get_final_verdict(dom_emotion_label, blink_rate, dom_posture, dom_audio)

    summary_data = {
        "Dominant Audio Emotion": dom_audio,
        "Dominant Posture": f"{dom_posture} ({upright_pct:.1f}%)",
        "Blink Rate": f"{blink_rate:.1f} blinks/min",
        "Dominant Emotion (Visual)": f"{dom_emotion_label} ({dom_emotion})",
        "Final Verdict": verdict,
        "Emotion Scores": emo_means.round(2).to_dict()
    }

    # Save to PDF
    pdf_path = "session_summary.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    for key, value in summary_data.items():
        if key == "Emotion Scores":
            c.drawString(50, y, "Emotion Scores:")
            y -= 20
            for e, v in value.items():
                c.drawString(70, y, f"{e}: {v}")
                y -= 20
        else:
            c.drawString(50, y, f"{key}: {value}")
            y -= 25
    c.save()
    print("PDF saved at:", pdf_path)

    return summary_data, pdf_path
# Streamlit UI
st.title("🎥Multimodal Real-Time Emotion Analyzer")
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

class Processor(VideoProcessorBase):
    def __init__(self):
        self.fno = 0
        self.blink = BlinkDetector()
        self.posture = PostureDetector()
        self.emotion = EmotionDetector()
        self.frame_log = {
            "timestamp": [],
            "posture": [],
            "audio_emotion": [],
        }
        for e in self.emotion.EMOS:
            self.frame_log[e] = []
        self.start_time = time.time()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blink_result = self.blink.update(gray)
        if blink_result:
            ear, _ = blink_result
            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.fno += 1
        if self.fno % PROCESS_EVERY_N == 0:
            ts = round(time.time() - self.start_time, 2)
            post = self.posture.classify(img)
            emo = self.emotion.scores(img)
            emo_audio = get_current_audio_emotion()

            self.frame_log["timestamp"].append(ts)
            self.frame_log["posture"].append(post)
            self.frame_log["audio_emotion"].append(emo_audio)
            for e in self.emotion.EMOS:
                self.frame_log[e].append(emo[e])

        audio_emotion = get_current_audio_emotion()
        cv2.putText(img, f"AudioEmotion: {audio_emotion}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, f"Blinks: {self.blink.total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return VideoFrame.from_ndarray(img, format="bgr24")



ctx = webrtc_streamer(
    key = "analyzer",
    video_processor_factory=Processor,
    async_processing=True
)
if ctx.video_processor:
    st.session_state.processor = ctx.video_processor


if st.button("🔚 End Session & Show Summary"):
    if "processor" not in st.session_state:
        st.error("No processor running.")
    else:
        processor = st.session_state.processor
        summary, pdf_path = generate_summary_and_pdf(processor)
        if summary:
            st.session_state.summary_data = summary
            st.session_state.pdf_path = pdf_path
        else:
            st.error("⚠️ No session data recorded yet. Please let the camera run for a few seconds.")

if st.session_state.summary_data:
    st.subheader("Live Session Summary")
    for k, v in st.session_state.summary_data.items():
        if k == "Emotion Scores":
            st.subheader("Average Emotion Scores")
            st.dataframe(pd.DataFrame.from_dict(v, orient='index', columns=["Score"]))
        else:
            st.write(f"**{k}:** {v}") 

    if "pdf_path" in st.session_state and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button("Download PDF Summary", f, file_name="session_summary.pdf")

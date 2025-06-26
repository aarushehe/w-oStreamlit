import cv2, time, pandas as pd
from blink_detector    import BlinkDetector
from posture_detector  import PostureDetector
from emotion_detector  import EmotionDetector
import threading
import csv
from audio_analysis import start_audio_threads, get_current_audio_emotion

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
conf_em = ["neutral", "surprise", "angry", "happy"]
anx_em = ["fear", "sad", "disgust"]

def save_dictlist_csv(data, filename):
    if not data:
        print(f"No data to write to {filename}")
        return
    with open(filename, 'w', newline='') as f:
        if isinstance(data, list):
            filenames = data[0].keys()
        else:
            fieldnames = data.keys()
            data = [dict(zip(fieldnames, values)) for values in zip(*data.values())]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

OUTPUT_FRAME_CSV = "session_log.csv"
OUTPUT_BLINK_CSV = "blink_log.csv"
PROCESS_EVERY_N  = 5

blink   = BlinkDetector()
posture = PostureDetector()
emotion = EmotionDetector()

start_audio_threads()

frame_log = {"timestamp": [], "posture": [], "audio_emotion": []}
for e in emotion.EMOS: frame_log[e] = []
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
start = time.time()
frame_no = 0

print("Press q to quit …")
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blink_result = blink.update(gray)
    if blink_result:
        ear, _ = blink_result
        cv2.putText(frame, f"EAR:{ear:.2f}", (300,30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)


    frame_no += 1
    if frame_no % PROCESS_EVERY_N == 0:
        ts   = round(time.time() - start, 2)
        post = posture.classify(frame)
        emo  = emotion.scores(frame)
        emo_audio = get_current_audio_emotion()

        frame_log["timestamp"].append(ts)
        frame_log["posture"].append(post)
        frame_log["audio_emotion"].append(emo_audio)
        for e in emotion.EMOS:
            frame_log[e].append(emo[e])
    audio_emotion = get_current_audio_emotion()
    cv2.putText(frame, f"AudioEmotion: {audio_emotion}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Blinks:{blink.total}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Live Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

save_dictlist_csv(frame_log,  OUTPUT_FRAME_CSV)
save_dictlist_csv(blink.log,   OUTPUT_BLINK_CSV)
print(f"Saved {len(frame_log['timestamp'])} rows → {OUTPUT_FRAME_CSV}")
print(f"Saved {len(blink.log)} blinks → {OUTPUT_BLINK_CSV}")


# === SESSION SUMMARY ===
df = pd.DataFrame(frame_log)

# Posture summary
from collections import Counter
p_counts = Counter(df["posture"])
upright_pct = p_counts.get("Upright", 0) / len(df) * 100 if len(df) else 0
dom_posture = "Upright" if upright_pct >= 20 else "Slouched"

# Emotion summary
emo_means = df[emotion.EMOS].mean()
conf_score = sum(emo_means[e] for e in conf_em)
anx_score  = sum(emo_means[e] for e in anx_em)
dom_emotion = "Confident" if conf_score >= anx_score else "Anxious"

# Blink summary
total_blinks = blink.total
duration_minutes = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 60
blink_rate = total_blinks / duration_minutes if duration_minutes > 0 else 0

# Final verdict based on dominant emotion + blink + posture
final_state = dom_emotion
if blink_rate > 30 or dom_posture == "Slouched":
    final_state = "Anxious"

# === DISPLAY SUMMARY ===
print("\n=== LIVE SESSION SUMMARY ===")
print("Posture counts       :", dict(p_counts))
print(f"Upright %            : {upright_pct:.1f}%")
print(f"Dominant posture     : {dom_posture}")
print("Average emotions     :")
for e in emotion.EMOS:
    print(f"  {e:<9}: {emo_means[e]:.2f}")
print(f"Confident score      : {conf_score:.2f}")
print(f"Anxious score        : {anx_score:.2f}")
print(f"Blink rate (blinks/min): {blink_rate:.1f}")
print(f"👉 FINAL VERDICT      : {final_state}")

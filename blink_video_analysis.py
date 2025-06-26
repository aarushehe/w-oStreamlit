import cv2
import time
import mediapipe as mp
import numpy as np

# === Video Input ===
video_path = "input.mp4"  
cap = cv2.VideoCapture(video_path)

# === MediaPipe FaceMesh Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === Eye Landmark Indices ===
LEFT_EYE = [33, 159, 158, 133, 153, 144]
RIGHT_EYE = [362, 386, 385, 263, 380, 373]

# === EAR Function ===
def eye_aspect_ratio(landmarks, eye_indices, shape):
    h, w = shape
    points = [landmarks[i] for i in eye_indices]
    coords = [(int(p.x * w), int(p.y * h)) for p in points]
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (v1 + v2) / (2.0 * h1)

# === Blink Detection Setup ===
ear_threshold = 0.22
blink_count = 0
blink_start = None
timestamps = []
fps = cap.get(cv2.CAP_PROP_FPS)

print("[INFO] Analyzing blinks...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, frame.shape[:2])
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, frame.shape[:2])
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < ear_threshold:
            if blink_start is None:
                blink_start = time.time()
        else:
            if blink_start is not None:
                duration = time.time() - blink_start
                if 0.1 < duration < 0.5:
                    blink_count += 1
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # Seconds
                blink_start = None

cap.release()
print(f"[RESULT] Total blinks detected: {blink_count}")

# === Infer Psychological State ===
duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
blink_rate = blink_count / duration_sec * 60  # Blinks per minute

print(f"[INFO] Blink rate: {blink_rate:.2f} blinks/min")

if blink_rate > 25:
    state = "Anxious"
elif blink_rate < 10:
    state = "Confident"
else:
    state = "Neutral/Normal"

print(f"[INFERENCE] The person appears to be: {state}")

# === Save blink timestamps ===
with open("video_blink_log.txt", "w") as f:
    for t in timestamps:
        f.write(f"{t:.2f} sec\n")

print("[INFO] Blink timestamps saved to 'video_blink_log.txt'")

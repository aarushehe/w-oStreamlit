from scipy.spatial import distance as dist
from imutils import face_utils
import cv2, dlib, numpy as np, os
import time
import pandas as pd

MODEL_PATH = "model/shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5
blink_log = []
start_time = time.time()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Cannot find {MODEL_PATH}. Make sure the .dat file is in the model/ folder.")

print("[INFO] loading facial landmark predictor...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

COUNTER = 0
TOTAL   = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)         
while True:
    ok, frame = cap.read()
    if not ok: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for rect in detector(gray, 0):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                timestamp = round(time.time() - start_time, 2)
                blink_log.append({"Blink #": TOTAL, "Timestamp (s)": timestamp})
            COUNTER = 0

        cv2.putText(frame, f"Blinks: {TOTAL}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0,255,0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)
        
    cv2.imshow("Blink detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if blink_log:
    df = pd.DataFrame(blink_log)
    df.to_csv("blink_log.csv", index=False)
    print("\n Blink log saved to blink_log.csv")
else:
    print("\n No blinks detected.")
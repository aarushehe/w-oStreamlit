# blink_detector.py
import time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils

MODEL_PATH = "model/shape_predictor_68_face_landmarks.dat"

class BlinkDetector:
    def __init__(self,
                 eye_thresh=0.23,
                 consec_frames=3,
                 debounce=0.4):
        self.detector   = dlib.get_frontal_face_detector()
        self.predictor  = dlib.shape_predictor(MODEL_PATH)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.EYE_THR    = eye_thresh
        self.MIN_CONSEC = consec_frames
        self.DEBOUNCE   = debounce

        self.counter    = 0
        self.total      = 0
        self.last_time  = 0
        self.start_time = time.time()
        self.log        = []      # list of dicts

    @staticmethod
    def _ear(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def update(self, gray):
        rects = self.detector(gray, 0)
        if not rects:
            return None   # no face

        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        left  = shape[self.lStart:self.lEnd]
        right = shape[self.rStart:self.rEnd]
        ear   = (self._ear(left) + self._ear(right)) / 2.0

        blink_happened = False
        if ear < self.EYE_THR:
            self.counter += 1
        else:
            if self.counter >= self.MIN_CONSEC:
                now = time.time()
                if now - self.last_time > self.DEBOUNCE:
                    self.total += 1
                    ts = round(now - self.start_time, 2)
                    self.log.append({"Blink #": self.total,
                                     "Timestamp (s)": ts})
                    blink_happened = True
                    self.last_time = now
            self.counter = 0
        return ear, blink_happened

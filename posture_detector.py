import cv2, math, mediapipe as mp

mp_pose = mp.solutions.pose

class PostureDetector:
    def __init__(self, angle_thresh = 140):
        self.pose = mp_pose.Pose()
        self.ANG_THR = angle_thresh

    @staticmethod
    def _angle(lm, ie, is_, ih, w, h):
        try:
            pts = [(lm[i].x*w, lm[i].y*h) for i in (ie, is_, ih) if lm[i].visibility > .5]
            if len(pts) < 3: return None
            a, b, c = pts
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            dot = ba[0]*bc[0] + ba[1]*bc[1]
            mag = math.hypot(*ba)*math.hypot(*bc)
            return math.degrees(math.acos(max(-1., min(1., dot/mag))))
        except:
            return None
    
    def classify(self, frame):
        h, w = frame.shape[:2]
        res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return "No person"
        lm = res.pose_landmarks.landmark
        L = self._angle(lm, mp_pose.PoseLandmark.LEFT_EAR.value,
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
        R = self._angle(lm, mp_pose.PoseLandmark.RIGHT_EAR.value,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
        valid = [a for a in (L, R) if a]
        if not valid: return "No person"
        return "Upright" if sum(valid)/len(valid) >= self.ANG_THR else "Slouched"
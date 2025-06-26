from deepface import DeepFace

class EmotionDetector:
    EMOS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        pass

    def scores(self, frame):
        zero = {e: 0.0 for e in self.EMOS}
        try:
            res = DeepFace.analyze(frame,
                                   actions=["emotion"],
                                   detector_backend="opencv",
                                   enforce_detection=False)
            raw = res[0] if isinstance(res, list) else res
            if "emotion" in raw:
                return raw["emotion"]
        except:
            pass
        return zero
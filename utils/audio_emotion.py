import librosa
import numpy as np

def extract_audio_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc = 13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    features = {
        "mfcc_mean": np.mean(mfccs),
        "chroma_mean": np.mean(chroma),
        "zcr_mean": np.mean(zcr),
        "rms_mean": np.mean(rms),
    }
    return features

def classify_emotion_from_audio(features):
    if features["rms_mean"] > 0.02 and features["mfcc_mean"] > -150:
        return "confident"
    else:
        return "anxious"
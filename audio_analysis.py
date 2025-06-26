import threading
from utils.speech_utils import speech_loop
from utils.breathing_utils import periodic_breath_analysis
from utils.audio_emotion import extract_audio_features, classify_emotion_from_audio
import pyaudio
import numpy as np
import time

AUDIO_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

def start_audio_threads():
    threading.Thread(target=speech_loop, daemon=True).start()
    threading.Thread(target=periodic_breath_analysis, daemon=True).start()

def analyze_audio_chunk(audio_buffer, sr=16000):
    features = extract_audio_features(audio_buffer, sr)
    emotion = classify_emotion_from_audio(features)
    print("Detected audio features:", emotion)
    return emotion 

detected_audio_emotion = "N/A"
_lock = threading.Lock()

def audio_stream():
    global detected_audio_emotion
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=CHUNK)
    buffer = []

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            buffer.extend(audio)

            if len(buffer) >= 2 * AUDIO_RATE:
                chunk = np.array(buffer[:2 * AUDIO_RATE])
                buffer = buffer[CHUNK:]

                features = extract_audio_features(chunk, AUDIO_RATE)
                emotion = classify_emotion_from_audio(features)

                with _lock:
                    detected_audio_emotion = emotion
        except Exception as e:
            print("Audio stream error:", e)
            break
    stream.stop_stream()
    stream.close()
    p.terminate()

def get_current_audio_emotion():
    with _lock:
        return detected_audio_emotion

def start_audio_threads():
    t = threading.Thread(target=audio_stream, daemon=True)
    t.start()
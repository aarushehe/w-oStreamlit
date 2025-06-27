import librosa, numpy as np, os, time

def detect_breath(audio_file):
    y, sr, = librosa.load(audio_file, sr=16000)
    energy = np.array([sum(abs(y[i:i+1024]**2)) for i in range(0, len(y), 1024)])
    threshold = np.percentile(energy, 10)
    breath_frames = energy < threshold
    breath_ratio = np.sum(breath_frames) / len(energy)

def periodic_breath_analysis(folder="dataset/audioFiles"):
    print("[Breathing detection Started]")
    while True:
        files = sorted(os.listdir(folder), reverse=True)
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                is_breathing = detect_breath(path)
                print("Breathing Detected " if is_breathing else "No breathing")
                break
        time.sleep(5)
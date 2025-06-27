import webrtcvad, pyaudio

def speech_loop():
    vad = webrtcvad.Vad(2)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    FRAME_DURATION = 30
    FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

    print("[Speech Detection Started]")
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        if vad.is_speech(frame, RATE):
            print("Speech Detected")
        else:
            print("Silence")
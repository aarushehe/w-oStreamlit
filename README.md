
# 🎥 Multimodal Real-Time Emotion Analyzer

A real-time emotion analyzer that uses your **webcam** and **microphone** to assess:
- Facial expressions (via DeepFace)
- Posture (via MediaPipe)
- Eye blinks (via EAR landmarks)
- Voice tone (via audio emotion classification)

At the end of the session, it:
- Generates a **PDF summary**
- Saves all logs to **MongoDB**

---

## 📌 Features

- 🎯 Real-time facial emotion recognition (DeepFace)
- 🧍 Posture detection (MediaPipe)
- 👁️ Blink rate detection (using eye aspect ratio)
- 🎤 Audio-based emotion classification (librosa + sklearn)
- 🧾 PDF session summary generation
- ☁️ Local MongoDB integration for session storage

---

## 💻 Installation Guide (Windows & macOS)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aarushehe/w-oStreamlit
cd w-oStreamlit
2️⃣ Create a Virtual Environment
🪟 For Windows
py -3.10 -m venv tf-env
.\tf-env\Scripts\activate
🍏 For macOS
python3 -m venv tf-env
source tf-env/bin/activate
3️⃣ Install Required Dependencies
🧠 Special pre-built wheels (Windows only)
Download and install:
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install webrtcvad_wheels-2.0.14-cp310-cp310-win_amd64.whl
On macOS, these can usually be installed via pip, but you may need Xcode command line tools for dlib.

✅ Install Python Requirements
pip install -r requirements.txt
🧠 Running the App
▶️ Option 1: Run Camera App (Terminal-based)
python camera_app.py
▶️ Option 2: Run Streamlit App (Browser-based GUI)
streamlit run streamlit.py
🗃️ MongoDB Setup (Local)
1️⃣ Download MongoDB Community Edition:
https://www.mongodb.com/try/download/community

Choose ZIP version if you don't have admin access.

2️⃣ Extract & Run MongoDB Locally
cd "path\to\mongodb\bin"      # Replace with your actual path
.\mongod.exe --dbpath ..\data\db    # For Windows
# macOS (after Homebrew installation)
brew services start mongodb-community
MongoDB must be running in the background before using the app.

📂 Output Files
session_log.csv — Full frame-level analysis log

blink_log.csv — Blink timestamps

session_summary.pdf — Downloadable PDF report

MongoDB — Stores session summary per run

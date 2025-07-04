# 🎥 Multimodal Real-Time Emotion Analyzer

A comprehensive real-time emotion analysis system that leverages your **webcam** and **microphone** to provide multi-dimensional emotional insights through:

- **Facial Expression Analysis** (DeepFace)
- **Posture Detection** (MediaPipe)
- **Blink Rate Monitoring** (Eye Aspect Ratio)
- **Voice Tone Classification** (Audio ML)

## ✨ Key Features

| Feature | Technology | Description |
|---------|------------|-------------|
| 🎯 **Facial Emotions** | DeepFace | Real-time facial emotion recognition |
| 🧍 **Posture Analysis** | MediaPipe | Body pose and posture detection |
| 👁️ **Blink Detection** | EAR Landmarks | Eye blink rate monitoring |
| 🎤 **Voice Analysis** | Librosa + Sklearn | Audio-based emotion classification |
| 📄 **PDF Reports** | ReportLab | Automated session summaries |
| 🗄️ **Data Storage** | MongoDB | Local session data persistence |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam and microphone access
- MongoDB (for data storage)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/aarushehe/w-oStreamlit
cd w-oStreamlit
```

#### 2. Set Up Virtual Environment

**Windows:**
```bash
py -3.10 -m venv tf-env
.\tf-env\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv tf-env
source tf-env/bin/activate
```

#### 3. Install Dependencies

**Windows (Special Requirements):**
```bash
# Install pre-built wheels first
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install webrtcvad_wheels-2.0.14-cp310-cp310-win_amd64.whl

# Install remaining requirements
pip install tf-keras
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# May require Xcode command line tools for dlib
pip install -r requirements.txt
```

## 🎮 Usage

### Option 1: Terminal Interface
```bash
python camera_app.py
```

### Option 2: Web Interface (Recommended)
```bash
streamlit run final.py
```
Then open your browser to `http://localhost:8501`

## 🗄️ MongoDB Setup

### Windows
1. Download [MongoDB Community Edition](https://www.mongodb.com/try/download/community)
2. Extract the ZIP file
3. Run MongoDB:
```bash
cd "path\to\mongodb\bin"
.\mongod.exe --dbpath ..\data\db
```

### macOS (with Homebrew)
```bash
brew install mongodb-community
brew services start mongodb-community
```

### Docker (Alternative)
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

> **⚠️ Important:** MongoDB must be running before starting the application.

## 📊 Output Files

The application generates several output files:

| File | Description |
|------|-------------|
| `session_log.csv` | Detailed frame-by-frame analysis |
| `blink_log.csv` | Timestamped blink events |
| `session_summary.pdf` | Comprehensive session report |
| MongoDB Collection | Persistent session storage |

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Ensure no other applications are using the camera

**MongoDB connection failed:**
- Verify MongoDB is running
- Check connection string in configuration

**Missing dependencies:**
- For Windows: Ensure Visual C++ redistributables are installed
- For macOS: Install Xcode command line tools: `xcode-select --install`

## 📋 Requirements

### System Requirements
- **OS:** Windows 10+, macOS 10.14+, or Linux
- **Python:** 3.10 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 2GB free space

### Hardware Requirements
- Webcam (720p or higher recommended)
- Microphone
- Stable internet connection (for initial model downloads)

---

**Made by [aarushehe](https://github.com/aarushehe)**

> 💡 **Tip:** For the best experience, ensure good lighting and position yourself 2-3 feet from the camera.

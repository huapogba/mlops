# AI Crowd Detection & Tracking System

A real-time intelligent system for detecting and tracking crowds using advanced computer vision and deep learning. Built with **YOLOv8**, **FastAPI**, and **WebSocket** for production-ready deployment.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-red.svg)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [ML Pipeline](#ml-pipeline)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Server Deployment](#server-deployment)

---

## Features

### Core Capabilities

- **Real-time Object Detection** 
  - Detects people and heads with high accuracy
  - Confidence scoring for each detection
  - Adaptive confidence thresholds

- **Multi-Object Tracking** 
  - Unique ID assignment for each person
  - Trajectory tracking across frames
  - Total unique person counting
  - Real-time crowd statistics

- **Automatic Model Retraining** 
  - Monitors data distribution drift
  - Automatic model versioning
  - Performance-based decision making
  - Prevents model degradation

- **Real-time Monitoring** 
  - Live FPS monitoring
  - Inference latency tracking
  - Prometheus metrics integration
  - Grafana dashboards support

- **Web-based Interface** 
  - Live camera feed streaming
  - Image analysis upload
  - Video file processing
  - Historical detection logs
  - Real-time statistics display

- **Enterprise-Ready** 
  - Docker containerization
  - MySQL database persistence
  - MLflow experiment tracking
  - DVC data versioning
  - GPU support

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│              (HTML5 + CSS3 + JavaScript + WebSocket)            │
│  • Live camera stream viewer    • Image analyzer                │
│  • Video file processor         • Detection history             │
└────────────────────────────────┬────────────────────────────────┘
                                 │ WebSocket / REST API
┌────────────────────────────────▼────────────────────────────────┐
│                         Backend Layer                           │
│                      (FastAPI + Python)                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ • Frame Processing & Validation                            │ │
│  │ • YOLOv8 Inference                                         │ │
│  │ • ByteTrack / BotsORT Tracking                             │ │
│  │ • Metrics Calculation                                      │ │
│  │ • WebSocket Broadcasting                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────┘
                                 │
    ┌────────────────────────────┼──────────────────────────────┐
    │                            │                              │
┌───▼───────────────┐  ┌─────────▼────────────┐  ┌──────────────▼─┐
│  MySQL Database   │  │  ML Pipeline Layer   │  │   Monitoring   │
│                   │  │                      │  │                │
│ • Detection logs  │  │ • Auto-training      │  │ • Prometheus   │
│ • User sessions   │  │ • Drift detection    │  │ • MLflow       │
│ • Statistics      │  │ • Model versioning   │  │ • Grafana      │
│ • Camera config   │  │ • DVC integration    │  │                │
└───────────────────┘  └───────────────────── ┘  └────────────────┘
```

---

##  Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **AI/ML** | YOLOv8 | Latest |
| **Tracking** | ByteTrack / BotsORT | Latest |
| **Backend** | FastAPI | 0.104+ |
| **Server** | Uvicorn | 0.24+ |
| **Database** | MySQL | 8.0+ |
| **ORM** | SQLAlchemy | 2.0+ |
| **Data Processing** | Pandas, NumPy | Latest |
| **Computer Vision** | OpenCV | 4.8+ |
| **Deep Learning** | PyTorch | 2.0+ |
| **Monitoring** | Prometheus | 0.18+ |
| **Experiment Tracking** | MLflow | 2.0+ |
| **Data Versioning** | DVC | 3.0+ |
| **Containerization** | Docker | Latest |
| **Frontend** | HTML5, CSS3, JavaScript | - |

---

##  Project Structure

```
crowd/
├── backend/
│   ├── main.py                 # FastAPI application & WebSocket handlers
│   ├── yolo_model.py          # YOLODetector class
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation & metrics
│   ├── drift.py               # Data drift detection
│   ├── prepare_data.py        # Data preparation from DB
│   ├── decide.py              # Model decision logic
│   ├── watcher.py             # Automatic trigger for retraining
│   ├── auto_train.py          # Auto-training loop
│   ├── bytetrack_custom.yaml  # Tracking configuration
│   └── model/                 # Model storage
│
├── frontend/
│   ├── main.html              # Web interface
│   ├── main.css               # Styling
│   └── main.js                # Client-side logic
│
├── data/
│   └── versions/              # Versioned datasets
│       ├── latest/
│       ├── v1/
│       └── v2/
│
├── model/
│   ├── best.pt                # Best ONNX model
│   ├── best.onnx              # ONNX format
│   └── best.engine            # TensorRT engine
│
├── mlruns/                    # MLflow artifacts
├── runs/                      # Training outputs
├── prometheus/                # Prometheus configuration
│
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Hyperparameters
├── dataset.yaml               # Dataset configuration
├── docker-compose.yaml        # Docker orchestration
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Application container
└── README.md                  # This file
```

---

##  Installation

### Prerequisites

- **Python 3.8+**
- **Docker & Docker Compose** (recommended)
- **GPU** (NVIDIA CUDA 11.8+, recommended for performance)
- **4GB+ RAM**
- **20GB+ Storage** (for models and datasets)

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/huapogba/mlops.git
cd mlops

# Build and start services
docker-compose up -d

# Check services
docker-compose ps
```

Services will be available at:
- **Backend API**: http://localhost:8000
- **MLflow**: http://localhost:5005
- **Prometheus**: http://localhost:9090
- **MySQL**: localhost:3307

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/huapogba/mlops.git
cd mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (optional, auto-downloaded on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Run backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In another terminal, run watcher (for auto-training)
python backend/watcher.py
```

---

##  Quick Start

### 1. Start the System

```bash
# Using Docker
docker-compose up -d

# Or locally
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 2. Access the Interface

Open your browser: **http://localhost:8000/static/main.html**

### 3. Add a Camera

In the **" Camera Live"** tab:
```
Camera Name: Main Gate
RTSP URL: rtsp://camera-ip:554/stream
```

### 4. Analyze an Image

Upload an image in **" Ảnh phân tích"** tab to get instant detection results.

### 5. Process a Video

Upload a video file in **" Video File"** tab for batch processing.

---

##  Configuration

### Hyperparameters (`params.yaml`)

```yaml
thresholds:
  score_bad: 0.4      # Poor model performance
  score_ok: 0.6       # Acceptable performance
```

### Environment Variables

Create `.env` file:

```env
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/smart_camera
MLFLOW_URI=http://localhost:5000
YOLO_CONF_THRESHOLD=0.4
YOLO_IMG_SIZE=640
TRACKER_TYPE=botsort
```

### Model Configuration

In `backend/yolo_model.py`:

```python
YOLODetector(
    model_path="yolov8n.pt",      # Model size: n, s, m, l, x
    conf_threshold=0.4,            # Confidence threshold
    img_size=832,                  # Input image size
    tracker="botsort.yaml"         # Tracking algorithm
)
```

---

##  Usage

### API Endpoints

#### 1. **Real-time Detection (WebSocket)**

```javascript
// JavaScript client
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Current crowd count:', data.people_count);
    console.log('FPS:', data.fps);
};
```

#### 2. **Image Upload & Detection**

```bash
curl -X POST "http://localhost:8000/api/detect/image" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 200, 400],
      "id": 1
    }
  ],
  "total_count": 1,
  "inference_time": 0.045
}
```

#### 3. **Video Analysis**

```bash
curl -X POST "http://localhost:8000/api/detect/video" \
  -F "file=@video.mp4"
```

#### 4. **Detection History**

```bash
curl "http://localhost:8000/api/history?camera_name=Main%20Gate&limit=100"
```

---

##  ML Pipeline

### Data Collection Flow

```
Raw Video Stream
    ↓
Frame Extraction (backend/main.py)
    ↓
Image Storage (static/history)
    ↓
Database Logging (MySQL)
    ↓
Monitor (backend/watcher.py)
    ↓
When new_files >= 100:
    ├→ Prepare Data (backend/prepare_data.py)
    ├→ Check Drift (backend/drift.py)
    └→ If drift > threshold: Train (DVC pipeline)
```

### Auto-Training Pipeline

1. **Data Preparation** (`backend/prepare_data.py`)
   - Load detections from database
   - Split into train/val sets
   - Generate YOLO-format labels
   - Version dataset with DVC

2. **Model Training** (`backend/train.py`)
   - Fine-tune YOLOv8 on new dataset
   - Log hyperparameters & metrics to MLflow
   - Track training with DVC
   - Save best checkpoint

3. **Evaluation** (`backend/evaluate.py`)
   - Validate on test set
   - Calculate mAP, precision, recall
   - Compare with baseline model
   - Log metrics to MLflow

4. **Decision** (`backend/decide.py`)
   - Compare performance metrics
   - Decide: accept, fine-tune, or retrain
   - Save decision to `decision.txt`

5. **Monitoring** (`backend/watcher.py`)
   - Watch for new data continuously
   - Trigger drift detection
   - Auto-trigger retraining if needed

### Running Manual Training

```bash
# Trigger DVC pipeline
dvc repro -f

# Or run individual steps
python backend/prepare_data.py
python backend/drift.py
python backend/train.py
python backend/evaluate.py
python backend/decide.py
```

---

##  Monitoring

### MLflow - Experiment Tracking

Access at: **http://localhost:5005**

Features:
- Track all training experiments
- Compare model performance
- Manage model versions
- View hyperparameters & metrics

```bash
# View experiments in CLI
mlflow experiments list

# Compare runs
mlflow runs compare <run1_id> <run2_id>
```

### Prometheus Metrics

Access at: **http://localhost:9090**

Key metrics:
- `inference_requests_total` - Total inference count
- `inference_seconds` - Inference latency
- `camera_fps` - Real-time FPS
- `people_count` - Current detected people
- `tracked_people_count` - Tracked individuals

### DVC Pipeline Tracking

```bash
# View pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Compare experiments
dvc exp show
```

---

##  Troubleshooting

### Issue: GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Model Out of Memory

```python
# Reduce image size
img_size = 416  # Default is 640

# Use smaller model
model_size = "n"  # nano (or s, m, l, x)

# Reduce batch size
batch_size = 8  # Default is 16
```

### Issue: Database Connection Failed

```bash
# Check MySQL is running
docker-compose ps

# Restart MySQL
docker-compose restart mysql

# Verify connection string
echo $DATABASE_URL
```

### Issue: WebSocket Connection Refused

```bash
# Check backend is running
curl http://localhost:8000/docs

# Verify firewall allows port 8000
netstat -an | grep 8000
```

---

##  Key Performance Metrics

| Metric | YOLOv8s | 
|--------|---------|
| **mAP50** | 77.2% | 
| **mAP50_90** | 48.7% |
| **Recall** | 68.3% |
| **Precision** | 86.1% |

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
---

## Server Deployment

### Deployment Architecture

```text
Internet
    │
    ▼
Ubuntu Server (GPU)
    │
    ├── Frontend Container
    ├── FastAPI Container
    ├── MySQL Container
    ├── Prometheus Container
    ├── Grafana Container
    └── YOLOv8 Inference Service
```

### Deploy using Docker Compose

Build and start all services:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

Stop services:

```bash
docker compose down
```

### Deployment Benefits

- One-command deployment.
- Easy service management.
- Reproducible environment.
- Supports GPU-enabled inference.
- Easy scaling and maintenance.
- ---

## Practical Course Extension

### Additional Work for Practical Course

The original theoretical project focused on developing and evaluating the AI crowd detection model.

For the practical course, the system was extended with deployment and operational capabilities:

- Containerization using Docker.
- Service orchestration using Docker Compose.
- Deployment on Ubuntu Server.
- Monitoring with Prometheus and Grafana.
- Production-ready architecture for real-world operation.

### Main Contribution

The main contribution of the practical course is transforming the AI prototype into a deployable system that can run continuously on a server environment with monitoring and service management capabilities.

##  Author

**AI Crowd Detection Team**

---

##  Acknowledgments

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [ByteTrack by Zhang et al.](https://github.com/ifzhang/ByteTrack)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)

---

##  Support

For questions or issues:
- Open an [Issue](https://github.com/yourusername/crowd-detection/issues)
- Check [Documentation](https://github.com/yourusername/crowd-detection/wiki)
- Contact: your-email@example.com

---

## If this project helped you, please give it a star!

# OmniNav iPhone Inference Setup

This repository contains the essential files for running OmniNav iPhone inference.

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on GB10, aarch64)
- 16GB+ GPU memory recommended

### Large Files (Download Separately)
These files are too large for GitHub and must be downloaded separately:

| File | Size | Download |
|------|------|----------|
| Model Weights | 7.7GB | [Google Drive Link - TBD] |

Place the model weights in: `models/chongchongjj/OmniNav/`

---

## 🚀 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/ayunlee/OmniNav_setup.git
cd OmniNav_setup
```

### 2. Build Docker Image (~30-60 minutes)
```bash
docker build -f Dockerfile.aarch64 -t omninav:aarch64 .
```

### 3. Download Model Weights
Download from Google Drive and place in `models/chongchongjj/OmniNav/`

### 4. Prepare iPhone Data
```bash
# Create data directory
mkdir -p data/iphone/<DATA_ID>

# Required files:
# - rgb.mp4 (video)
# - odometry.csv (pose data)
# - instruction.txt (navigation instruction)
# - camera_matrix.csv
# - imu.csv
# - depth/ (depth images)
# - confidence/ (confidence maps)

# Extract frames (~50 frames recommended)
cd data/iphone/<DATA_ID>
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 rgb.mp4
# Calculate interval: total_frames / 50

mkdir -p rgb
ffmpeg -i rgb.mp4 -vf "select=not(mod(n\,<INTERVAL>))" -vsync vfr -q:v 2 rgb/%06d.png
```

### 5. Run Inference
```bash
docker run --gpus all --rm \
  -v $(pwd):/workspace/OmniNav \
  --shm-size=8gb \
  -w /workspace/OmniNav/infer_r2r_rxr \
  omninav:aarch64 \
  python3 run_infer_iphone.py \
    --data-dir ../data/iphone/<DATA_ID> \
    --model-path ../models/chongchongjj/OmniNav \
    --result-path ../data/result_<DATA_ID>
```

---

## 📁 Repository Structure

```
OmniNav_setup/
├── Dockerfile.aarch64          # Docker build file
├── README.md                   # This file
├── docs/
│   └── iphone_inference_guide.md  # Detailed guide
├── infer_r2r_rxr/
│   ├── run_infer_iphone.py     # Main inference script
│   ├── agent/
│   │   └── waypoint_agent.py   # Model loading & inference
│   └── VLN_CE/                 # Habitat extensions
└── train_code/
    └── transformers-main/      # Custom Transformers (Qwen2.5-VL)
```

---

## 📊 Output

Results are saved in:
```
data/result_<DATA_ID>/
└── models/chongchongjj/OmniNav/
    ├── log/inference_*.log     # Inference logs
    └── map_vis/<DATA_ID>/      # Visualization images
```

---

## 🔍 Troubleshooting

### Check GPU
```bash
nvidia-smi
```

### Monitor Logs
```bash
tail -f data/result_*/models/chongchongjj/OmniNav/log/inference_*.log
```

### Check Docker Container
```bash
docker ps
```

---

## 📖 Documentation

See [docs/iphone_inference_guide.md](docs/iphone_inference_guide.md) for detailed instructions.



Walk forward towards the black chair in front of you. Once you reach the chair, make a left turn. Then, continue walking straight, following the white table on your right side. When you see the end of the white table, make a right turn. Approach the blue door in front of you, get close to it, and stop to finalize the trajectory.

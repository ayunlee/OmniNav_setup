# OmniNav iPhone 인퍼런스 실행 가이드

## 📋 개요

이 가이드는 iPhone에서 촬영한 데이터를 사용하여 OmniNav 모델 인퍼런스를 실행하는 방법을 설명합니다.

> [!IMPORTANT]
> **좌표 정렬(Umeyama) 및 3방향 크롭 기능**을 사용하려면 `docs/iphone_inference_cropped_guide.md`를 참조하고, `run_infer_iphone_cropped.py` 스크립트를 사용하세요. 아래 가이드의 `run_infer_iphone.py`는 기본 스크립트로, 좌표 정렬 기능을 지원하지 않을 수 있습니다.

---

## 📐 데이터셋별 변환 파라미터 (참고용)

`run_infer_iphone_cropped.py` 사용 시 필요한 데이터셋별 Umeyama 변환 파라미터입니다.

| 데이터 ID | Scale | Rotation | Translation |
|---|---|---|---|
| `1f9786888a` | 3.65 | 160.05° | [1.29, 1.88] |
| `a1af0cece0` | 67.24 | -4.15° | [-0.96, 3.10] |
| `441dc05e1e` | 3.79 | 108.62° | [0.32, -0.76] |

---

## 📁 필수 데이터 구조

새로운 데이터를 추가할 때, 다음과 같은 구조로 파일을 준비해야 합니다:

```
data/iphone/<데이터_ID>/
├── rgb.mp4              # 영상 파일
├── odometry.csv         # 위치/자세 정보
├── imu.csv              # IMU 센서 데이터
├── camera_matrix.csv    # 카메라 내부 파라미터
├── instruction.txt      # 네비게이션 지시문
├── depth/               # 깊이 이미지 폴더
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── confidence/          # 신뢰도 이미지 폴더
    ├── 0000.png
    ├── 0001.png
    └── ...
```

---

## 🚀 인퍼런스 실행 방법

### 기본 실행 명령어

```bash
cd /home/aprl_msi/OmniNav_20260115_giseopkim

OUTPUT_DIR="result_iphone_$(date +%Y%m%d_%H%M%S)" && \
LOG_FILE="/tmp/iphone_inference_r2r_$(date +%Y%m%d_%H%M%S).log" && \
echo "출력 디렉토리: $OUTPUT_DIR" && \
echo "로그 파일: $LOG_FILE" && \
docker run --gpus all --rm \
  -v /home/aprl_msi/OmniNav_20260115_giseopkim:/workspace/OmniNav \
  --shm-size=8gb \
  -w /workspace/OmniNav/infer_r2r_rxr \
  omninav:aarch64 \
  python3 run_infer_iphone.py \
    --data-dir ../data/iphone/<데이터_ID> \
    --model-path ../models/chongchongjj/OmniNav \
    --result-path ../data/$OUTPUT_DIR \
    --max-frames 0 \
  > "$LOG_FILE" 2>&1 &

echo "프로세스 ID: $!"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "로그 파일: $LOG_FILE"
```

### 예시: ca9e389640 데이터 실행

```bash
cd /home/aprl_msi/OmniNav_20260115_giseopkim

OUTPUT_DIR="result_iphone_$(date +%Y%m%d_%H%M%S)" && \
LOG_FILE="/tmp/iphone_inference_r2r_$(date +%Y%m%d_%H%M%S).log" && \
docker run --gpus all --rm \
  -v /home/aprl_msi/OmniNav_20260115_giseopkim:/workspace/OmniNav \
  --shm-size=8gb \
  -w /workspace/OmniNav/infer_r2r_rxr \
  omninav:aarch64 \
  python3 run_infer_iphone.py \
    --data-dir ../data/iphone/ca9e389640 \
    --model-path ../models/chongchongjj/OmniNav \
    --result-path ../data/$OUTPUT_DIR \
    --max-frames 0 \
  > "$LOG_FILE" 2>&1 &
```

---

## 📌 명령어 옵션 설명

| 옵션 | 설명 | 예시 값 |
|------|------|---------|
| `--data-dir` | 입력 데이터 경로 (컨테이너 내부 기준) | `../data/iphone/ca9e389640` |
| `--model-path` | OmniNav 모델 경로 | `../models/chongchongjj/OmniNav` |
| `--result-path` | 결과 저장 경로 | `../data/result_iphone_YYYYMMDD_HHMMSS` |
| `--max-frames` | 처리할 최대 프레임 수 (0 = 전체) | `0`, `100`, `500` |

---

## 📊 실시간 로그 확인

### 임시 로그 파일 확인
```bash
tail -f /tmp/iphone_inference_r2r_YYYYMMDD_HHMMSS.log
```

### 결과 디렉토리 내 상세 로그 확인
```bash
tail -f data/result_iphone_YYYYMMDD_HHMMSS/models/chongchongjj/OmniNav/log/inference_*.log
```

---

## 📂 결과물 위치

인퍼런스 완료 후 결과물은 다음 위치에 저장됩니다:

```
data/result_iphone_YYYYMMDD_HHMMSS/
└── models/
    └── chongchongjj/
        └── OmniNav/
            └── log/
                ├── inference_*.log    # 상세 로그
                └── stats_*.json       # 통계 및 결과
```

---

## 🔧 Docker 관련 참고사항

### Docker 이미지
- 사용 이미지: `omninav:aarch64`

### GPU 설정
- `--gpus all`: 모든 GPU 사용
- `--shm-size=8gb`: 공유 메모리 크기 (모델 로딩에 필요)

### 볼륨 마운트
- 호스트의 `/home/aprl_msi/OmniNav_20260115_giseopkim`이 컨테이너 내부 `/workspace/OmniNav`에 마운트됨

---

## ⚡ 빠른 실행 원라이너

새 데이터 `<데이터_ID>`로 빠르게 실행하려면:

```bash
DATA_ID="<데이터_ID>" && cd /home/aprl_msi/OmniNav_20260115_giseopkim && OUTPUT_DIR="result_iphone_$(date +%Y%m%d_%H%M%S)" && docker run --gpus all --rm -v /home/aprl_msi/OmniNav_20260115_giseopkim:/workspace/OmniNav --shm-size=8gb -w /workspace/OmniNav/infer_r2r_rxr omninav:aarch64 python3 run_infer_iphone.py --data-dir ../data/iphone/$DATA_ID --model-path ../models/chongchongjj/OmniNav --result-path ../data/$OUTPUT_DIR --max-frames 0 > "/tmp/inference_${DATA_ID}.log" 2>&1 & echo "PID: $! | 출력: $OUTPUT_DIR | 로그: /tmp/inference_${DATA_ID}.log"
```

**예시:**
```bash
DATA_ID="ca9e389640" && cd /home/aprl_msi/OmniNav_20260115_giseopkim && OUTPUT_DIR="result_iphone_$(date +%Y%m%d_%H%M%S)" && docker run --gpus all --rm -v /home/aprl_msi/OmniNav_20260115_giseopkim:/workspace/OmniNav --shm-size=8gb -w /workspace/OmniNav/infer_r2r_rxr omninav:aarch64 python3 run_infer_iphone.py --data-dir ../data/iphone/$DATA_ID --model-path ../models/chongchongjj/OmniNav --result-path ../data/$OUTPUT_DIR --max-frames 0 > "/tmp/inference_${DATA_ID}.log" 2>&1 & echo "PID: $! | 출력: $OUTPUT_DIR | 로그: /tmp/inference_${DATA_ID}.log"
```

---

## 🔍 문제 해결

### 프로세스 상태 확인
```bash
ps aux | grep run_infer_iphone.py
```

### Docker 컨테이너 확인
```bash
docker ps
```

### GPU 사용량 확인
```bash
nvidia-smi
```

---

## 🎬 RGB 프레임 추출 (필수 전처리)

새 데이터에 `rgb/` 폴더가 없는 경우, 먼저 `rgb.mp4`에서 프레임을 추출해야 합니다.

### ⭐ 권장: 프레임 간격 조절 (50-60개 프레임)

용량 절약을 위해 **모든 프레임 대신 일정 간격으로 50-60개만 추출**하는 것을 권장합니다:

```bash
# 데이터 디렉토리로 이동
cd /home/aprl_msi/OmniNav_20260115_giseopkim/data/iphone/<데이터_ID>

# 1. 먼저 영상의 총 프레임 수 확인
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 rgb.mp4

# 2. 적절한 간격으로 프레임 추출 (예: 매 5번째 프레임만 추출)
mkdir -p rgb && ffmpeg -i rgb.mp4 -vf "select=not(mod(n\,5))" -vsync vfr -q:v 2 rgb/%06d.png
```

**프레임 간격 계산:**
- 총 프레임 수 / 목표 프레임 수 = 간격
- 예: 300개 프레임 → 60개 원하면 → 간격 5 (`mod(n\,5)`)
- 예: 600개 프레임 → 60개 원하면 → 간격 10 (`mod(n\,10)`)

| 총 프레임 수 | 목표 | 간격 설정 |
|-------------|------|----------|
| ~300 | 50-60개 | `mod(n\,5)` |
| ~600 | 50-60개 | `mod(n\,10)` |
| ~900 | 50-60개 | `mod(n\,15)` |

**예시 (ca9e389640, 293개 프레임 → 59개로 축소):**
```bash
cd /home/aprl_msi/OmniNav_20260115_giseopkim/data/iphone/ca9e389640
mkdir -p rgb && ffmpeg -i rgb.mp4 -vf "select=not(mod(n\,5))" -vsync vfr -q:v 2 rgb/%06d.png
```

### 전체 프레임 추출 (비권장 - 용량 주의)

모든 프레임을 추출하려면:
```bash
mkdir -p rgb && ffmpeg -i rgb.mp4 -q:v 2 rgb/%06d.png
```

> [!WARNING]
> 전체 프레임 추출 시 용량이 매우 커집니다 (수백 MB ~ 수 GB).
> 인퍼런스 시간도 프레임 수에 비례하여 증가합니다.

> [!IMPORTANT]
> `rgb/` 폴더가 없으면 인퍼런스에서 0개의 프레임이 처리됩니다!

---

## 📝 체크리스트

새 데이터로 인퍼런스 실행 전 확인사항:

- [ ] `rgb.mp4` 파일 존재
- [ ] `rgb/` 폴더 존재 (없으면 위 "RGB 프레임 추출" 섹션 참고)
- [ ] `odometry.csv` 파일 존재
- [ ] `imu.csv` 파일 존재  
- [ ] `camera_matrix.csv` 파일 존재
- [ ] `instruction.txt` 파일 존재 (네비게이션 지시문)
- [ ] `depth/` 폴더 및 깊이 이미지들 존재
- [ ] `confidence/` 폴더 및 신뢰도 이미지들 존재
- [ ] Docker 이미지 `omninav:aarch64` 사용 가능

---

## 📈 결과 해석 (Visualization & Logs)

### 1. `map_vis` 이미지 해석 (노란색 궤도)

`map_vis` 폴더에 생성되는 이미지(또는 GIF)는 로봇의 시야와 예측된 지도를 보여줍니다.

- **노란색 선/점 (Yellow Trajectory)**: **모델이 예측한 이동 경로 (Predicted Waypoints)**입니다.
  - 현재 위치에서 로봇이 어디로 가려고 하는지를 나타냅니다.
- **검은색 (Black)**: (데이터셋에 GT가 있는 경우) 정답 경로 (Ground Truth)입니다. iPhone 데이터 같은 커스텀 데이터에는 없을 수 있습니다.
- **회색 영역**: 갈 수 있는 공간 (Navigable area)으로 인식된 부분입니다.

### 2. `arrive_pred` (도착 예측 값) 확인 방법

`arrive_pred`는 로봇이 목적지에 도착했다고 판단하는 확률(Logit 또는 0~1 값)입니다.

**방법 A: 로그 파일 확인**
실시간 로그나 저장된 로그 파일에서 `arrive=` 항목을 확인하세요.
```text
Frame 82: arrive=0, heading[0]=18.02°, wp[0]=(0.0012, 0.0400)
```
- `arrive=0`: 아직 도착하지 않음 (계속 이동)
- `arrive=1` (또는 > 0.5): 도착했다고 판단 (STOP 액션 수행 예정)

**방법 B: 이미지 하단 텍스트**
`map_vis`의 각 프레임 이미지 하단에 텍스트로 정보가 기록됩니다.
- 형식: `..., arrive_pred_[값], ...`
- 여기서 `[값]`이 구체적인 모델의 도착 예측 수치(Raw output)입니다. 이 값이 특정 임계값(0.5 등)을 넘으면 로봇이 멈춥니다.


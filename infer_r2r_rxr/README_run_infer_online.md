# run_infer_online.py

OmniNav 실시간 추론 스크립트. ROS2 환경에서 이미지를 입력받아 웨이포인트를 예측하고 `/action` 토픽으로 퍼블리시한다.

---

## 동작 모드

### 1. `--data-dir` 지정 시 (Panorama 모드)

- **이미지 구독 없음.** 카메라 토픽을 구독하지 않는다.
- `data_dir/instruction.txt`와 `data_dir/rgb/` 아래 **프레임 폴더만** 사용한다.
- 루프에서 프레임 폴더를 순회하며, 각 프레임마다 **실제 front / left / right** 이미지를 디스크에서 읽어 inference 한다.
- 예측 웨이포인트는 기존처럼 **`/action`에 퍼블리시** 한다.
- 폴더·파일 구조는 `run_infer_iphone_panorama.py`와 동일하다.

### 2. `--data-dir` 없을 때 (기존 모드)

- **한 개 이미지 토픽 구독**을 유지한다. (기본: `/camera/camera/color/image_raw/compressed`)
- 받은 이미지를 **front**로 쓰고, **left / right**는 front 복사(기존과 동일)로 사용한다.
- 예측 웨이포인트는 `/action`에 퍼블리시한다.

---

## Panorama 모드 데이터 구조

`run_infer_iphone_panorama.py`와 같은 구조:

```
data_dir/
├── instruction.txt
└── rgb/
    ├── frame_0000/
    │   ├── frame_0000_front.jpg
    │   ├── frame_0000_left.jpg
    │   └── frame_0000_right.jpg
    ├── frame_0001/
    │   ├── frame_0001_front.jpg
    │   ├── frame_0001_left.jpg
    │   └── frame_0001_right.jpg
    └── ...
```

---

## 사용법

### Panorama 모드 (front/left/right 디스크에서 로드)

```bash
python run_infer_online.py \
  --model-path /path/to/OmniNav \
  --data-dir /path/to/data/Test18 \
  --result-path ./results
```

- `--instruction`은 사용하지 않는다. instruction은 `data_dir/instruction.txt`에서 읽는다.

### 기존 모드 (한 개 토픽 구독, left/right = front 복사)

```bash
python run_infer_online.py \
  --model-path /path/to/OmniNav \
  --instruction "Go to the kitchen" \
  --result-path ./results
```

### 공통 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model-path` | OmniNav 모델 경로 | (필수) |
| `--instruction` | 네비게이션 지시문 (data-dir 미사용 시) | `""` |
| `--data-dir` | Panorama 구조 데이터 디렉터리 | `None` |
| `--result-path` | 결과 저장 경로 | `./results` |
| `--inference-interval` | 추론 간격(초) | `1.0` |
| `--save-video` | 시각화 영상 저장 | `True` |
| `--no-save-video` | 영상 저장 비활성화 | - |

---

## 출력

- **CSV**: `result_path/waypoint_data_online_{timestamp}.csv`  
  - 컬럼: `frame_idx`, `subframe_idx`, `dx`, `dy`, `dtheta`, `arrive`, `infer_time_s`
- **영상** (저장 활성화 시): `result_path/omninav_online_{timestamp}.mp4`  
  - 프론트 뷰에 웨이포인트 화살표가 그려진 시각화

---

## 요약

| 모드 | 이미지 소스 | front | left / right |
|------|-------------|--------|------------------|
| **Panorama** (`--data-dir` 지정) | 디스크 | `frame_XXXX_front.jpg` | `frame_XXXX_left.jpg`, `frame_XXXX_right.jpg` |
| **기존** (`--data-dir` 없음) | ROS 토픽 1개 | 구독 이미지 | front 복사 |

두 모드 모두 예측 웨이포인트는 `/action` 토픽으로 퍼블리시된다.

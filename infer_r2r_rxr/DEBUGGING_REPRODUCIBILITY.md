# 🔍 환경 간 추론 결과 차이 디버깅 가이드

## 문제 상황
- 환경 A와 환경 B에서 같은 데이터, 같은 모델 파라미터로 추론했는데 결과가 다름
- 패키지 버전 차이가 있음

## 📦 현재 패키지 버전 차이

### 환경 A (내 환경)
- `torch`: 2.9.0a0+50eac811a6.nv25.09 (nightly/development 버전)
- `opencv-python`: 4.11.0
- `Pillow`: 11.3.0
- `scipy`: 1.16.1
- `safetensors`: 0.6.2

### 환경 B (다른 시스템)
- `torch`: 2.6.0 (stable)
- `opencv-python`: 4.10.0
- `Pillow`: 11.1.0
- `scipy`: 1.14.1
- `safetensors`: 0.5.2

---

## 🎯 의심해볼 포인트 (우선순위 순)

### 1. ⚠️ **PyTorch 버전 차이 (가장 중요!)**

**문제:**
- 환경 A는 **PyTorch 2.9.0a0** (nightly/development 버전)
- 환경 B는 **PyTorch 2.6.0** (stable 버전)
- Nightly 버전은 불안정하고 연산 결과가 다를 수 있음

**확인 방법:**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
```

**해결책:**
- **환경 A를 PyTorch 2.6.0 stable 버전으로 다운그레이드** (권장)
- 또는 환경 B를 2.9.0a0으로 업그레이드 (비권장, 불안정)

**PyTorch 버전별 연산 차이:**
- `torch.bfloat16` 연산 정밀도 차이
- Attention 메커니즘 구현 차이
- cuDNN 알고리즘 선택 차이

---

### 2. 🔢 **NumPy 버전 차이**

**확인 필요:**
- NumPy 버전이 다르면 부동소수점 연산 결과가 달라질 수 있음
- 특히 `np.float32` vs `np.float64` 기본 타입 차이

**확인 방법:**
```python
import numpy as np
print(f"NumPy: {np.__version__}")
print(f"Default float type: {np.array([1.0]).dtype}")
```

**해결책:**
- 두 환경의 NumPy 버전을 동일하게 맞춤

---

### 3. 🖼️ **이미지 로딩/전처리 차이 (OpenCV, Pillow)**

**문제:**
- `opencv-python`: 4.11.0 vs 4.10.0
- `Pillow`: 11.3.0 vs 11.1.0
- 이미지 디코딩, 리사이징 알고리즘이 다를 수 있음

**확인 방법:**
```python
import cv2
from PIL import Image
print(f"OpenCV: {cv2.__version__}")
print(f"Pillow: {Image.__version__}")

# 이미지 로딩 테스트
img1 = cv2.imread("test.jpg")
img2 = Image.open("test.jpg")
print(f"OpenCV shape: {img1.shape}, dtype: {img1.dtype}")
print(f"Pillow mode: {img2.mode}, size: {img2.size}")
```

**해결책:**
- OpenCV와 Pillow 버전을 동일하게 맞춤
- 이미지 전처리 파이프라인에서 명시적으로 interpolation 방법 지정

---

### 4. 🎲 **난수 시드 미설정**

**문제:**
- 코드에 명시적인 시드 설정이 없음
- Dropout, Attention mask 등에서 난수 사용 시 결과가 달라질 수 있음

**확인 방법:**
```python
# run_infer_iphone_panorama.py에 시드 설정이 있는지 확인
grep -r "seed\|random\|manual_seed" infer_r2r_rxr/
```

**해결책:**
- 추론 시작 전에 시드 고정:
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

### 5. 🚀 **CUDA/cuDNN 버전 차이**

**문제:**
- GPU 드라이버, CUDA 버전, cuDNN 버전이 다르면 연산 결과가 달라질 수 있음
- 특히 cuDNN은 여러 알고리즘 중 선택하는데, 버전에 따라 선택이 다를 수 있음

**확인 방법:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**해결책:**
- CUDA/cuDNN 버전을 동일하게 맞추기 어려우면, deterministic 모드 활성화:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

### 6. 🐍 **Python 버전 차이**

**확인 필요:**
- Python 버전이 다르면 부동소수점 연산 결과가 달라질 수 있음

**확인 방법:**
```python
import sys
print(f"Python: {sys.version}")
```

**해결책:**
- Python 버전을 동일하게 맞춤 (권장: 3.8-3.11)

---

### 7. 📚 **기타 의존성 패키지 버전 차이**

**확인 필요:**
- `transformers` 버전
- `qwen-vl-utils` 버전
- `safetensors` 버전 (0.6.2 vs 0.5.2 - 모델 로딩 방식 차이 가능)

**확인 방법:**
```python
import transformers
import qwen_vl_utils
import safetensors
print(f"transformers: {transformers.__version__}")
print(f"qwen-vl-utils: {qwen_vl_utils.__version__}")
print(f"safetensors: {safetensors.__version__}")
```

**해결책:**
- 모든 의존성 패키지 버전을 동일하게 맞춤

---

## 🔧 권장 해결 순서

### Step 1: PyTorch 버전 통일 (최우선)
```bash
# 환경 A에서 PyTorch 2.6.0으로 다운그레이드
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: 시드 고정 코드 추가
`run_infer_iphone_panorama.py` 파일 상단에 추가:
```python
# 시드 고정 (재현성 보장)
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Step 3: 패키지 버전 통일
두 환경의 모든 패키지 버전을 동일하게 맞춤:
```bash
# requirements.txt 생성
pip freeze > requirements.txt

# 다른 환경에서 설치
pip install -r requirements.txt
```

### Step 4: CUDA/cuDNN 확인
두 환경의 CUDA/cuDNN 버전 확인 및 가능하면 통일

---

## 📊 결과 비교 방법

### 1. 중간 결과 비교
- 각 프레임의 waypoint 출력값 비교
- 모델의 중간 레이어 출력 비교

### 2. 통계 비교
- 전체 경로의 평균/표준편차 비교
- 최종 목표 도달률 비교

### 3. 시각화 비교
- 예측 경로 시각화 비교
- 각 프레임의 attention map 비교 (가능하면)

---

## ⚠️ 주의사항

1. **PyTorch nightly 버전 사용 금지**: 프로덕션/실험에서는 stable 버전만 사용
2. **부동소수점 연산의 한계**: 완전히 동일한 결과는 어려울 수 있음 (1e-6 수준의 차이는 정상)
3. **GPU 하드웨어 차이**: 다른 GPU 모델을 사용하면 연산 결과가 다를 수 있음

---

## 🎯 빠른 체크리스트

- [ ] PyTorch 버전 통일 (2.6.0 stable 권장)
- [ ] NumPy 버전 통일
- [ ] OpenCV 버전 통일
- [ ] Pillow 버전 통일
- [ ] 시드 고정 코드 추가
- [ ] CUDA/cuDNN 버전 확인
- [ ] Python 버전 통일
- [ ] transformers 버전 통일
- [ ] safetensors 버전 통일
- [ ] 모든 의존성 패키지 버전 통일

---

## 📝 참고

- PyTorch 버전별 호환성: https://pytorch.org/get-started/previous-versions/
- cuDNN deterministic 모드: https://pytorch.org/docs/stable/notes/randomness.html


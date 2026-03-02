# 패키지 판단 기준 (Package Detection Criteria)

이 문서는 `check_package_versions.py` 스크립트가 `run_infer_iphone_panorama.py` 실행에 필요한 패키지를 어떻게 판단했는지 설명합니다.

## 판단 기준

### 1. 직접 Import 분석

`run_infer_iphone_panorama.py` 파일에서 직접 import된 외부 패키지:

```python
import numpy as np          # numpy
import torch               # torch (PyTorch)
from tqdm import trange    # tqdm
import cv2                 # opencv-python
from PIL import Image      # Pillow
```

**표준 라이브러리 (제외)**:
- `os`, `sys`, `argparse`, `csv`, `json`, `time`, `pathlib`, `datetime` → Python 표준 라이브러리이므로 제외

### 2. 간접 Import 분석 (의존성 체인)

`run_infer_iphone_panorama.py`가 import하는 `agent.waypoint_agent` 모듈에서 사용하는 패키지:

```python
# waypoint_agent.py에서 import된 외부 패키지:
import numpy as np                    # numpy (중복)
import torch                          # torch (중복)
from tqdm import trange               # tqdm (중복)
import cv2                            # opencv-python (중복)
from PIL import Image                 # Pillow (중복)
from scipy.spatial.transform import Rotation as R  # scipy
from safetensors.torch import load_file            # safetensors
from transformers import AutoProcessor, AutoTokenizer, ...  # transformers
from qwen_vl_utils import process_vision_info      # qwen_vl_utils
```

**선택적 패키지 (try-except로 처리됨)**:
```python
try:
    import imageio                    # imageio (GIF 저장용, 선택적)
    from habitat import Env           # habitat (시뮬레이터용, 선택적)
except:
    pass
```

### 3. 최종 패키지 목록

#### 필수 패키지 (9개)
1. **numpy** - 수치 연산
2. **torch** - PyTorch (딥러닝 프레임워크)
3. **tqdm** - 진행 표시줄
4. **opencv-python** - 이미지 처리 (cv2)
5. **Pillow** - 이미지 처리 (PIL)
6. **transformers** - Hugging Face Transformers (Qwen 모델 로딩)
7. **qwen_vl_utils** - Qwen Vision-Language 유틸리티
8. **scipy** - 과학 계산 (회전 변환 등)
9. **safetensors** - 모델 가중치 로딩

#### 선택적 패키지 (2개)
10. **imageio** - GIF 저장용 (없어도 동작하지만 GIF 저장 불가)
11. **habitat** - Habitat 시뮬레이터 (iPhone 데이터 사용 시 불필요)

## 판단 로직

1. **표준 라이브러리 제외**: Python에 기본 포함된 모듈은 제외
2. **의존성 체인 추적**: 직접 import뿐만 아니라 간접적으로 사용되는 패키지도 포함
3. **실제 사용 여부 확인**: import 문이 실제로 실행되는지 확인 (try-except 제외)
4. **선택적 패키지 구분**: try-except로 감싸진 패키지는 선택적으로 표시

## 검증 방법

스크립트는 다음 방법으로 패키지 존재 여부를 확인합니다:

1. **`__import__()` 사용**: 실제로 모듈을 import하여 존재 여부 확인
2. **버전 정보 추출**: `__version__`, `version`, `VERSION` 속성 확인
3. **Import 테스트**: 실제 import 문을 실행하여 동작 여부 확인

## 참고

- 표준 라이브러리와 외부 패키지의 구분은 Python 공식 문서를 기준으로 함
- 패키지 이름과 import 이름이 다른 경우 (예: `opencv-python` → `cv2`) 별도로 매핑
- Docker 컨테이너 환경의 LD_PRELOAD 설정도 고려하여 스크립트에 포함


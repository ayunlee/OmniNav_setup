# 패키지 다운그레이드 의존성 분석 결과

## 📋 다운그레이드 계획

| Package | 현재 버전 | 목표 버전 | 상태 |
|---------|----------|----------|------|
| torch | 2.9.0a0+50eac811a6.nv25.09 | 2.6.0 | ⚠️ 확인 필요 |
| opencv-python | 4.11.0.86 | 4.10.0 | ❌ 버전 없음 |
| Pillow | 11.3.0 | 11.1.0 | ✅ 가능 |
| scipy | 1.16.1 | 1.14.1 | ✅ 가능 |
| safetensors | 0.6.2 | 0.5.2 | ✅ 가능 |

## 🔍 의존성 분석

### 1. transformers 요구사항
- **safetensors**: `>=0.4.3` 요구
- **numpy**: `>=1.17` 요구
- **tqdm**: `>=4.27` 요구

### 2. qwen-vl-utils 요구사항
- **pillow**: 요구 (구체적 버전 없음)
- **packaging**: 요구
- **requests**: 요구

### 3. torch 의존성
- torch 2.6.0은 사용 가능한 버전임
- 하지만 NVIDIA GB10 (DGX Spark)에서 CUDA 호환성 확인 필요

## ⚠️ 발견된 문제

### 1. opencv-python 버전 이슈
```
ERROR: Could not find a version that satisfies the requirement opencv-python==4.10.0
```
**해결책**: 
- `4.10.0.82` 또는 `4.10.0.84` 사용 (마이너 버전 포함)
- 또는 `4.10.0.*` 사용

### 2. safetensors 버전 확인
- transformers는 `safetensors>=0.4.3` 요구
- 목표 버전 `0.5.2`는 `>=0.4.3`을 만족하므로 ✅ OK

### 3. torch 다운그레이드 주의사항
- 현재: `2.9.0a0+50eac811a6.nv25.09` (NVIDIA 최적화 빌드)
- 목표: `2.6.0` (일반 PyPI 버전)
- **주의**: NVIDIA 최적화 빌드에서 일반 빌드로 변경 시 성능 차이 가능

## ✅ 안전한 다운그레이드 순서

1. **safetensors**: `0.6.2` → `0.5.2` ✅
2. **Pillow**: `11.3.0` → `11.1.0` ✅
3. **scipy**: `1.16.1` → `1.14.1` ✅
4. **opencv-python**: `4.11.0.86` → `4.10.0.82` (또는 `4.10.0.84`) ⚠️
5. **torch**: `2.9.0a0+...` → `2.6.0` ⚠️ (가장 마지막에, 테스트 필수)

## 🧪 테스트 권장사항

다운그레이드 후 다음을 테스트:
1. `python3 check_package_versions.py` - 패키지 버전 확인
2. `python3 -c "from agent.waypoint_agent import Waypoint_Agent"` - Import 테스트
3. 실제 inference 실행 테스트

## 📝 권장 명령어

```bash
# 1. safetensors
pip install safetensors==0.5.2

# 2. Pillow
pip install Pillow==11.1.0

# 3. scipy
pip install scipy==1.14.1

# 4. opencv-python (정확한 버전 사용)
pip install opencv-python==4.10.0.82

# 5. torch (마지막에, 주의 깊게)
pip install torch==2.6.0

# 테스트
python3 check_package_versions.py
python3 -c "from agent.waypoint_agent import Waypoint_Agent; print('OK')"
```


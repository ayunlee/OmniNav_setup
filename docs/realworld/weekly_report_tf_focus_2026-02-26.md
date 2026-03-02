# OmniNav 실세계 연동 주간보고 (TF 중심)

작성일: 2026-02-26  
작성자: APRL OmniNav 팀

---

## 1) 이번 주 목표

OmniNav slow+fast 실세계 데모를 위해, FAST-LIO2 + Scout Mini + Orbbec 3Cam 입력 체인을 정리하고, 특히 Phase 2의 TF/시간동기 기반을 안정화하는 것을 목표로 진행했습니다.

---

## 2) 진행 내용 요약 (무엇을 위해 무엇을 했는지)

### 2-1. 실세계 입력 파이프라인 구동을 위해 런치 구조를 분리했다

- `ros2 launch`가 블로킹이므로 컨테이너/터미널을 분리해 실행하도록 정리했습니다.
- 운영 기준:
  - `omninav_slow`: Livox 드라이버, 3카메라, Scout base, TF 브리지
  - `fastlio_container`: FAST-LIO2

### 2-2. 반복 소싱 부담을 줄이기 위해 컨테이너 자동 소싱을 구성했다

- 두 컨테이너(`omninav_slow`, `fastlio_container`)의 `/root/.bashrc`, `/root/.profile`에 ROS/워크스페이스 자동 소싱 블록을 추가했습니다.
- 목적: `docker exec -it <container> bash` 진입 후 즉시 명령 실행 가능하도록 정리.
- 검증: `bash -lc` 환경에서도 `ros2`와 패키지 인식 확인.

### 2-3. TF 체인 구성을 위해 Phase 2 스크립트를 보강했다

- `scout_tf_config/launch/sensor_tf_launch.py`에서 아래 TF 퍼블리싱을 한 번에 실행하도록 구성했습니다.
  - `base_link -> lidar_frame`
  - `lidar_frame -> cam_front/left/right_color_optical_frame`
  - `/Odometry` 기반 `map -> base_link` 브리지
  - `map -> camera_init` 정적 TF(FAST-LIO cloud frame 연결용)
- `scout_tf_config/scripts/verify_tf.py`에 `map -> camera_init` 연결 검증을 포함했습니다.

### 2-4. 실측 기반 TF 보정을 위해 파라미터 파일을 업데이트했다

- `scout_tf_config/config/sensor_tf.yaml`에 실측값 반영:
  - `base_link -> lidar_frame`: `x=0.09, y=0.0, z=0.242`
  - `lidar -> cam_*` translation은 줄자값으로 임시 반영
- 카메라 회전은 기존 캘리브레이션 결과를 유지하고, 위치는 현장 실측으로 우선 맞추는 전략으로 진행했습니다.

---

## 3) 주요 이슈와 대응

### 3-1. TF 트리가 분리되는 문제가 있었다

- 문제: `map`과 `camera_init`이 연결되지 않아 RViz에서 cloud가 `Fixed Frame=map` 기준으로 보이지 않는 케이스 발생.
- 대응: `map -> camera_init` 정적 TF를 런처에서 명시적으로 퍼블리시하도록 수정.
- 결과: `tf2_echo map camera_init` 관점에서 연결성 문제는 해소 방향으로 정리됨.

### 3-2. 카메라 translation이 실제 하드웨어 위치와 다르게 보이는 문제가 있었다

- 문제: 회전은 대체로 맞지만, 카메라 위치가 base/lidar 기준으로 산개하거나 높이가 어긋남.
- 대응:
  - `T_cam_lidar` vs `T_lidar_cam` 방향성 검토
  - 역행렬 테스트 후, 현장 실측 translation으로 임시 고정
- 현재 판단: 회전 정합은 비교적 양호, translation은 추가 미세조정 필요.

### 3-3. timestamp 허용치 경계에서 fail이 발생했다

- 문제: `verify_tf.py` 결과에서 `odom-cam` 델타가 100ms 기준을 가끔 초과.
- 예시 관측값: `0.097s / 0.107s / 0.120s`.
- 대응: 토픽 주파수/지터 재측정 및 장시간 안정성 테스트 필요 항목으로 분리.

### 3-4. FAST-LIO2 입력 필드/파라미터 관련 불안정 이슈가 있었다

- 문제:
  - `Failed to find match for field 'reflectivity'` 경고 발생 이력
  - `No point, skip this scan!`, `No Effective Points!` 반복 이력
- 대응:
  - Mid360 파서 수정 실험 및 재빌드까지 수행했으나, 요청에 따라 원상복구 완료
  - 현재는 원본 코드 기준으로 유지
- 현재 상태: FAST-LIO 안정 동작 조건을 다시 원본 기준에서 정리해야 함.

---

## 4) 실제 검증 결과 (현재까지)

### 4-1. 토픽/센서

- `/livox/lidar`: 약 10 Hz 수신 확인
- `/livox/imu`: 약 200 Hz 수신 확인
- `/Odometry`: 수신 확인
- `/cloud_registered`: 수신 확인

### 4-2. TF 검증

- `verify_tf.py` 기준으로 주요 체인 조회는 수행 가능
- 다만 timestamp 기준(<=100ms)은 환경/시점에 따라 경계값 초과가 관측되어, Gate G2는 “부분 통과” 상태

### 4-3. RViz 정합

- cloud 누적 및 odom 추종은 확인됨
- 카메라 translation은 “약간 어긋난 상태”로 판단되며 추가 보정 필요

---

## 5) Gate 관점 현재 상태

- Phase 2-1 (TF 체인 구성): 대부분 완료
- Phase 2-2 (시간동기/지연 안정성): 진행 중
- Phase 2-3 (RViz 정합 최종): 진행 중
- 종합: Gate G2 “미완료(추가 검증 필요)”

---

## 6) 다음 주(다음 작업) 계획

### 6-1. TF translation 미세보정 완료

- `lidar -> cam_*` translation 재측정/재적용
- RViz 기준으로 base/lidar/cam 상대 위치 정합 완료

### 6-2. 시간동기 장시간 테스트 수행

- 동일 런치 조건으로 10분 연속 구동
- `/Odometry`, `/cloud_registered`, `/cam_*` 드롭/지연 계측 로그 확보
- `RGB-POSE <= 100ms` 기준 통과 여부 재판정

### 6-3. FAST-LIO 원본 설정 기준 안정화

- 원복 상태에서 경고(`reflectivity`, `No Effective Points`) 재현 조건 정리
- `mid360.yaml` 파라미터(특히 `preprocess.lidar_type`) 최종 확인
- 안정 런치 레시피를 문서로 확정

### 6-4. Gate G2 통과 후 Phase 3 착수

- `lio_grid_node`, `rgb_memory_bank_node`, `habitat_adapter_node` 개발 시작
- slow frontier 선택 -> fast subgoal 전달까지 1차 폐루프 달성

---

## 7) 참고 파일

- `omninav_realworld_plan.md`
- `scout_tf_config/config/sensor_tf.yaml`
- `scout_tf_config/launch/sensor_tf_launch.py`
- `scout_tf_config/scripts/verify_tf.py`
- `docs/realworld/topic_io_contract_v1.md`


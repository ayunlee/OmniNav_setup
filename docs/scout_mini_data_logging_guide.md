# Scout Mini 데이터 로깅 운용 가이드

> OmniNav 추론이 아닌, **Scout Mini를 키보드로 조종하면서 카메라 데이터를 rosbag으로 로깅**하기 위한 가이드.

---

## 1. 시스템 구성 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                        이 PC (aprl_msi)                         │
│                                                                 │
│  [터미널 1] Scout Mini 베이스 드라이버  (CAN 통신)              │
│  [터미널 2] 카메라 드라이버             (USB)                   │
│  [터미널 3] 키보드 텔레옵               (WASD 조종)             │
│  [터미널 4] rosbag2 record             (데이터 저장)            │
└─────────────────────────────────────────────────────────────────┘
```

총 **4개 터미널**이 필요하며, 각각 독립적으로 실행합니다.

---

## 2. 필요한 파일 목록

프로젝트 루트: `/home/aprl_msi/OmniNav_setup/`

| 파일 | 역할 | 비고 |
|---|---|---|
| `scout_launch.sh` | Scout Mini 베이스 드라이버 런치 | CAN 인터페이스 설정 포함 |
| `scout_teleop.py` | WASD 키보드 텔레옵 스크립트 | ROS2 Python 노드 |
| `safe_triple_cam_launch.sh` | Orbbec Gemini 3대 런치 (래퍼) | USB 장치 확인 + 정리 로직 포함 |
| `triple_camera_launch.py` | Orbbec 3대 카메라 런치 파일 | 시리얼 번호로 front/left/right 구분 |
| `crop_node.py` | 이미지 리사이즈/크롭 노드 | 640x400 -> 480x426 변환 |
| `safe_realsense_launch.sh` | RealSense 단일 카메라 런치 (대안) | Orbbec 대신 RealSense 사용 시 |
| `gemini_driver/` | Orbbec 카메라 ROS2 드라이버 패키지 | `orbbec_camera` 패키지 포함 |

### ROS2 패키지 의존성 (빌드 필요)

| 패키지 | 위치 | 역할 |
|---|---|---|
| `scout_mini_base` | `install/scout_mini_base/` | Scout Mini CAN 드라이버 + base_launch.py |
| `scout_mini_description` | `install/scout_mini_description/` | URDF/모델 정보 |
| `scout_mini_msgs` | `install/scout_mini_msgs/` | Scout Mini 전용 메시지 타입 |
| `scout_mini_hardware` | `install/scout_mini_hardware/` | 하드웨어 인터페이스 |
| `orbbec_camera` | `gemini_driver/install/` | Orbbec 카메라 드라이버 |

---

## 3. 단계별 실행 방법

### 사전 조건

- Scout Mini 전원 ON, CAN 케이블 연결
- 카메라 USB 케이블 연결 (Orbbec 3대 또는 RealSense 1대)
- ROS2 (Jazzy) 환경이 설치되어 있을 것
- Scout Mini ROS2 패키지가 빌드되어 `install/` 디렉토리에 있을 것

---

### 터미널 1: Scout Mini 베이스 드라이버

```bash
cd /home/aprl_msi/OmniNav_setup
bash scout_launch.sh
```

**내부 동작:**
```bash
# 1. ROS2 패키지 환경 source
source ./install/setup.sh
source ./install/scout_mini_base/share/scout_mini_base/local_setup.bash
source ./install/scout_mini_description/share/scout_mini_description/local_setup.bash

# 2. CAN 인터페이스 활성화 (sudo 필요)
sudo ip link set can0 up type can bitrate 500000

# 3. 베이스 드라이버 런치
ros2 launch scout_mini_base base_launch.py
```

> [!IMPORTANT]
> CAN 인터페이스 설정에 sudo 권한이 필요합니다. 비밀번호를 입력하세요.

---

### 터미널 2: 카메라 드라이버

#### 옵션 A: Orbbec Gemini 3대 (front / left / right)

```bash
cd /home/aprl_msi/OmniNav_setup
bash safe_triple_cam_launch.sh
```

**생성되는 주요 토픽:**

| 토픽 | 설명 |
|---|---|
| `/cam_front/color/image_raw` | 전면 카메라 원본 (640x400) |
| `/cam_front/color/image_target` | 전면 카메라 크롭 (480x426) |
| `/cam_left/color/image_raw` | 좌측 카메라 원본 |
| `/cam_left/color/image_target` | 좌측 카메라 크롭 |
| `/cam_right/color/image_raw` | 우측 카메라 원본 |
| `/cam_right/color/image_target` | 우측 카메라 크롭 |

**카메라 시리얼 번호 매핑:**

| 위치 | 시리얼 | 런치 지연 |
|---|---|---|
| Front | `CP82841000G5` | 즉시 |
| Right | `CP82841000KH` | 4초 |
| Left | `CP82841000C2` | 8초 |

> [!NOTE]
> 카메라가 3대 미만 감지되면 경고가 뜨지만 강제 진행됩니다. USB 재연결 또는 PC 재부팅으로 해결하세요.

#### 옵션 B: RealSense 단일 카메라

```bash
cd /home/aprl_msi/OmniNav_setup
bash safe_realsense_launch.sh
```

**생성되는 주요 토픽:**

| 토픽 | 설명 |
|---|---|
| `/camera/camera/color/image_raw` | 컬러 이미지 (640x480, 5fps) |
| `/camera/camera/color/image_raw/compressed` | 압축 이미지 (플러그인 설치 시) |

---

### 터미널 3: 키보드 텔레옵 (조종)

```bash
cd /home/aprl_msi/OmniNav_setup
source ./install/setup.sh
python3 scout_teleop.py
```

**조작법:**

```
     [W]       전진
  [A][S][D]    좌회전 / 후진 / 우회전
   SPACE       긴급 정지
   CTRL-C      종료
```

**설정값 (스크립트 상단에서 조정 가능):**

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `LINEAR_SPEED` | 0.8 m/s | 직진 속도 |
| `ANGULAR_SPEED` | 0.8 rad/s | 회전 속도 |
| `POLLING_RATE` | 0.02s (50Hz) | 입력 감지 주기 |
| `KEY_PERSISTENCE` | 0.15s | 키 입력 유지 시간 (데드맨 스위치) |

**퍼블리시 토픽:** `/scout_mini_base_controller/cmd_vel` (TwistStamped, Best Effort QoS)

---

### 터미널 4: 데이터 로깅 (rosbag2)

```bash
# Orbbec 3대 사용 시 - 크롭된 이미지 저장
ros2 bag record \
  /cam_front/color/image_target \
  /cam_left/color/image_target \
  /cam_right/color/image_target \
  /scout_mini_base_controller/cmd_vel \
  -o ~/rosbag_data/session_$(date +%Y%m%d_%H%M%S)

# 또는 원본 이미지도 함께 저장
ros2 bag record \
  /cam_front/color/image_raw \
  /cam_front/color/image_target \
  /cam_left/color/image_raw \
  /cam_left/color/image_target \
  /cam_right/color/image_raw \
  /cam_right/color/image_target \
  /scout_mini_base_controller/cmd_vel \
  -o ~/rosbag_data/session_$(date +%Y%m%d_%H%M%S)
```

```bash
# RealSense 사용 시
ros2 bag record \
  /camera/camera/color/image_raw \
  /scout_mini_base_controller/cmd_vel \
  -o ~/rosbag_data/session_$(date +%Y%m%d_%H%M%S)
```

> [!TIP]
> 저장 용량이 크므로 `--max-bag-duration` 또는 `--max-bag-size` 옵션으로 분할 저장을 고려하세요.

---

## 4. 토픽 확인 및 디버깅

```bash
# 활성 토픽 목록
ros2 topic list

# 특정 토픽 데이터 확인
ros2 topic echo /cam_front/color/image_target --once

# 토픽 발행 주파수 확인
ros2 topic hz /cam_front/color/image_raw

# 이미지 시각화 (GUI 환경일 때)
ros2 run rqt_image_view rqt_image_view
```

---

## 5. 종료 순서

안전한 종료를 위해 **역순**으로 종료합니다:

1. **터미널 4** (rosbag): `Ctrl-C` -- 로그 저장 완료
2. **터미널 3** (텔레옵): `Ctrl-C` -- 로봇 자동 정지 명령 전송
3. **터미널 2** (카메라): `Ctrl-C` -- 카메라 드라이버 정리
4. **터미널 1** (베이스): `Ctrl-C` -- Scout Mini 드라이버 종료

---

## 6. 파일 구조 요약

```
OmniNav_setup/
├── scout_launch.sh              # [터미널 1] Scout Mini 베이스 드라이버
├── scout_teleop.py              # [터미널 3] WASD 키보드 조종
├── safe_triple_cam_launch.sh    # [터미널 2] Orbbec 3대 래퍼 스크립트
├── triple_camera_launch.py      # [터미널 2] Orbbec 3대 런치 파일
├── crop_node.py                 # [터미널 2] 이미지 리사이즈/크롭 노드
├── safe_realsense_launch.sh     # [터미널 2] RealSense 대안
├── gemini_driver/               # Orbbec 카메라 ROS2 드라이버
│   └── orbbec_camera/
│       └── launch/
│           └── gemini_330_series.launch.py
├── install/                     # 빌드된 ROS2 패키지들
│   ├── scout_mini_base/         # Scout Mini 드라이버
│   ├── scout_mini_description/  # URDF 모델
│   ├── scout_mini_msgs/         # 메시지 타입
│   └── scout_mini_hardware/     # 하드웨어 인터페이스
└── docs/
    └── scout_mini_data_logging_guide.md   # 이 문서
```

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `CAN interface not found` | CAN 케이블 미연결 또는 드라이버 문제 | 케이블 확인, `ip link show can0` 으로 상태 확인 |
| 텔레옵 키 입력이 안 먹힘 | 다른 노드가 cmd_vel을 점유 중 | 다른 퍼블리셔가 없는지 `ros2 topic info /scout_mini_base_controller/cmd_vel` 확인 |
| 카메라 3대 중 일부만 뜸 | USB 대역폭 부족 또는 장치 인식 실패 | USB 재연결, PC 재부팅, `lsusb \| grep Orbbec` 확인 |
| RealSense not detected | 카메라 미연결 또는 권한 문제 | USB 재연결, `rs-enumerate-devices` 명령 확인 |
| rosbag 용량이 너무 큼 | 원본 이미지 토픽 로깅 | `image_target` (크롭) 토픽만 로깅하거나 compressed 사용 |

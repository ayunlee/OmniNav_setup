# Phase 2: Static TF + TF 브리지 구성 요청 (v3)

## 목표
ROS2 Humble/Jazzy 환경에서 Scout Mini 로봇의 센서 TF 체인을 하나로 연결한다.

## 목표 TF 체인 (초기 데모 단순화)
```
map → base_link → lidar_frame → cam_front_color_optical_frame
                               → cam_left_color_optical_frame
                               → cam_right_color_optical_frame
```

> 참고: `map → odom → base_link` 분리는 후속 고도화 단계에서 추가 가능. Phase 2에서는 `map → base_link`로 고정.

## 현재 상태
- FAST-LIO2가 `/tf`로 `camera_init → body`를 발행 중 (단, timestamp가 sec:0 nanosec:0으로 찍히는 문제 있음)
- FAST-LIO2가 `/Odometry` 토픽도 발행 중 (정상 timestamp, ~10Hz)
- Scout 컨트롤러가 `odom → base_link`를 발행 중 (`enable_odom_tf: true`)
- 카메라 3대와 LiDAR의 static TF는 아직 없음
- 두 TF 트리가 분리되어 있음

## 작업 1: TF Authority 정리 — Scout odom TF 비활성화

FAST-LIO2를 위치 추정의 단일 authority로 사용한다.
Scout 컨트롤러의 odom TF 발행을 비활성화해야 한다.

Scout 설정 파일: `scout_mini.yaml`
```yaml
# 변경 필요:
enable_odom_tf: false   # true → false로 변경
```

**이유:** FAST-LIO와 Scout이 둘 다 base_link의 parent를 발행하면 다중 parent 충돌이 발생한다. FAST-LIO(LiDAR+IMU)가 Scout 휠 오도메트리보다 정확하므로 FAST-LIO를 기준으로 한다.

## 작업 2: FAST-LIO TF 브리지 노드

### 왜 브리지가 필요한가
1. FAST-LIO의 프레임명(`camera_init`, `body`)이 하드코딩되어 있어서 config로 변경 불가
2. FAST-LIO의 `/tf` 발행 timestamp가 0으로 찍혀서 tf2 lookup 시 문제 발생
3. topic remapping은 TF 메시지 내부의 frame_id 문자열에는 적용 안 됨

### 구현 요구사항
**`/Odometry` 토픽 기반 TF 브리지 노드**를 만들어줘:

- `/Odometry` (nav_msgs/Odometry) 토픽을 구독
- 해당 메시지의 pose를 읽어서 `map → base_link` TF를 발행
- `/Odometry`의 정상 timestamp를 그대로 사용 (sec:0 문제 회피)
- 프레임 매핑:
  - parent frame: `map` (= FAST-LIO의 camera_init 역할)
  - child frame: `base_link` (= FAST-LIO의 body 역할)

```python
# 핵심 로직 (참고)
# 1. /Odometry 구독
# 2. odom_msg.pose.pose에서 position, orientation 추출
# 3. TransformStamped 생성:
#    - header.stamp = odom_msg.header.stamp (원본 timestamp 유지)
#    - header.frame_id = "map"
#    - child_frame_id = "base_link"
# 4. tf_broadcaster.sendTransform(...)
```

**주의:** 이 브리지 노드가 실행되면 FAST-LIO의 원본 `camera_init → body` TF와 중복 발행될 수 있다. 둘이 공존해도 frame_id가 다르니 충돌은 안 나지만, 혼란 방지를 위해 FAST-LIO 원본 TF를 무시하거나 필터링하는 옵션도 고려해줘.

## 작업 3: Static TF 발행 launch 파일

### 3-1. lidar_frame → 카메라 3대 (캘리브레이션 데이터)

아래는 체커보드 캘리브레이션으로 구한 **LiDAR → Camera** 변환 4x4 행렬이다.
rotation(3x3) 부분을 쿼터니언으로 변환하고 translation을 사용해서 static TF를 발행해줘.

**⚠️ 행렬 방향 확인 필수:** 아래 행렬은 "LiDAR 좌표계 → Camera 좌표계" 변환이다. 즉 parent=lidar_frame, child=camera. 만약 반대(camera→lidar)라면 역행렬을 취해야 한다. 쿼터니언 변환 후 RViz에서 카메라 프레임 방향이 실제와 맞는지 반드시 확인할 것.

**Front Camera (lidar_frame → cam_front_color_optical_frame):**
```
A: [-0.0050   -1.0000   -0.0034    0.2226
    -0.0381    0.0036   -0.9993    0.1884
     0.9993   -0.0048   -0.0381    0.1304
          0         0         0    1.0000]
```
Translation: [0.2226, 0.1884, 0.1304]

**Left Camera (lidar_frame → cam_left_color_optical_frame):**
```
A: [ 0.9990   -0.0208   -0.0393    0.2811
    -0.0403   -0.0490   -0.9980    0.1470
     0.0188    0.9986   -0.0498   -0.0132
          0         0         0    1.0000]
```
Translation: [0.2811, 0.1470, -0.0132]

**Right Camera (lidar_frame → cam_right_color_optical_frame):**
```
A: [-0.9994    0.0198   -0.0293    0.0262
     0.0289   -0.0211   -0.9994    0.0462
    -0.0204   -0.9996    0.0205   -0.0973
          0         0         0    1.0000]
```
Translation: [0.0262, 0.0462, -0.0973]

### 3-2. base_link → lidar_frame (실측 반영)

```yaml
base_to_lidar:
  x: 0.09
  y: 0.0
  z: 0.242
  roll: 0.0
  pitch: 0.0
  yaw: 0.0
```

보정 근거:
- 줄자 실측: `ground → lidar = 0.43m`
- Scout URDF wheel geometry 기준: `base_link → ground ≈ 0.188m`
- 따라서 초기값: `base_link → lidar.z ≈ 0.43 - 0.188 = 0.242m`

주의:
- `z=0.43`를 그대로 넣으면 `base_link` 원점 정의와 충돌할 수 있음.
- RViz 정합 보고 수 cm 단위 미세조정 권장.

## 작업 4: 검증 스크립트

Phase 2 검증용 스크립트:

1. **TF 트리 확인:** `view_frames`로 단일 루트 트리 확인 (루프/다중 parent 없는지)
2. **TF lookup 테스트:** 아래 쌍이 모두 lookup 성공하는지 체크
   - `map → base_link`
   - `base_link → lidar_frame`
   - `lidar_frame → cam_front_color_optical_frame`
   - `lidar_frame → cam_left_color_optical_frame`
   - `lidar_frame → cam_right_color_optical_frame`
   - `map → cam_front_color_optical_frame` (전체 체인 관통 테스트)
3. **타임스탬프 오프셋 체크:** `/Odometry`와 `/cam_*/color/image_raw`의 header.stamp 차이가 100ms 이내인지
4. **Hz 체크:** 브리지 노드의 `map → base_link` TF 발행이 8Hz 이상인지

## 최종 파일 구조
```
scout_tf_config/
├── config/
│   └── sensor_tf.yaml          # 모든 TF 파라미터 (수정 용이)
├── launch/
│   └── sensor_tf_launch.py     # static TF + 브리지 노드 한번에 실행
├── scripts/
│   ├── odom_tf_bridge.py       # /Odometry → map→base_link TF 브리지
│   └── verify_tf.py            # 검증 스크립트
└── README.md
```

## 환경
- ROS2 Humble/Jazzy (실행 컨테이너 기준)
- Python launch 파일 사용
- Docker 컨테이너: `omninav_slow`

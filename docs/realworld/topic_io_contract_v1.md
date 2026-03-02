# Track A - Topic I/O Contract (V1, A-3)

작성일: 2026-02-24  
상태: 1차 확정 (Phase 3 통합에서 수정 가능)

## 1) 목적

Track A 분석 결과를 바탕으로, slow+fast 실세계 파이프라인의 토픽/메시지 계약을 1차 확정한다.

## 2) 확정 원칙

- `/action` 단일 발행자: `infer_r2r_rxr/run_infer_online_panorama.py`
- pose authority: FAST-LIO2 `/Odometry`
- slow/adapter는 `/action`을 직접 발행하지 않음
- adapter는 `/omni/subgoal`만 발행

## 3) 입력 토픽 (외부)

| Producer | Topic | Msg Type | 용도 |
|---|---|---|---|
| Livox driver | `/livox/lidar` | `livox_interfaces/msg/CustomMsg` 또는 `sensor_msgs/PointCloud2` | FAST-LIO 입력 |
| Livox/IMU | `/livox/imu` | `sensor_msgs/Imu` | FAST-LIO 입력 |
| FAST-LIO2 | `/cloud_registered` | `sensor_msgs/PointCloud2` | grid 생성 |
| FAST-LIO2 | `/Odometry` | `nav_msgs/Odometry` | pose 입력 |
| FAST-LIO2 | `/path` | `nav_msgs/Path` | 디버그/모니터링 |
| Orbbec Front | `/cam_front/color/image_raw` | `sensor_msgs/Image` | slow/fast 시각 입력 |
| Orbbec Left | `/cam_left/color/image_raw` | `sensor_msgs/Image` | slow/fast 시각 입력 |
| Orbbec Right | `/cam_right/color/image_raw` | `sensor_msgs/Image` | slow/fast 시각 입력 |

## 4) 내부 브리지 토픽 (신규)

| Producer | Topic | Msg Type (V1) | 용도 |
|---|---|---|---|
| `lio_grid_node` | `/omni/occupancy` | `nav_msgs/OccupancyGrid` | slow map 입력 |
| `lio_grid_node` | `/omni/pose2d` | `geometry_msgs/PoseStamped` | slow/fast pose 기준 |
| `habitat_adapter_node` | `/omni/frontiers` | `geometry_msgs/PoseArray` (임시) | frontier 후보 전달 |
| `habitat_adapter_node` | `/omni/subgoal` | `geometry_msgs/PoseStamped` | slow->fast handoff |
| `rgb_memory_bank_node` | `/omni/frontier_views` | 서비스/내부 API (V1) | frontier 대표 이미지 조회 |

참고:
- `/omni/frontiers` 타입은 V1 임시값이며, Phase 3에서 custom msg로 변경 가능
- 이미지 bank 조회는 토픽보다 서비스가 적합하므로 V1에서 서비스 우선

## 5) 실행 토픽 (기존 재사용)

| Producer | Topic | Msg Type | Consumer | 비고 |
|---|---|---|---|---|
| `run_infer_online_panorama.py` | `/action` | `std_msgs/String`(JSON payload) | `omninav_control.py` | 단일 발행자 |
| `omninav_control.py` | `/scout_mini_base_controller/cmd_vel` | `geometry_msgs/TwistStamped` | Scout base controller | 최종 제어 |

## 6) fast 입력 계약 (V1)

현재 코드 기준 (`infer_r2r_rxr/run_infer_online_panorama.py`):
- tri-RGB: `/cam_front|left|right/color/image_raw`
- instruction: CLI `--instruction`
- pose: 현재 `default_pose` 고정값 사용 중 (`run_infer_online_panorama.py:432`)

V1 요구사항:
- Phase 3에서 pose 입력을 `/Odometry` 또는 `/omni/pose2d`로 교체
- 모델 추론 로직은 유지, 입력 소스만 교체

## 7) Out of Scope (Track A)

- A* 세부 구현, snap policy, fallback policy
- frontier msg 정식 타입 확정
- local planner/trajectory smoothing 상세

위 항목은 Phase 3 구현 설계에서 확정한다.


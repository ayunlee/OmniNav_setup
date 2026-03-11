# Phase 3 상세 실행 계획 (OVON 로직 유지 기반 실세계 전환)

작성일: 2026-03-02  
대상: `OmniNav_setup`  
전제: Phase 1(치환표/토픽 계약) + Phase 2(TF/정합) 완료

## 1) 목표와 원칙

목표:
- OVON 데이터셋 기반 원래 slow+fast 의사결정 흐름을 최대한 유지하면서, Habitat 시뮬레이터 의존성을 ROS2 실센서 입력으로 치환한다.
- 최종 폐루프: `slow(subgoal)` -> `fast(/action)` -> `omninav_control(cmd_vel)` -> Scout Mini

고정 원칙:
- 모델/가중치/추론 핵심 로직 변경 금지
- `/action` 단일 발행자 유지 (`infer_r2r_rxr/run_infer_online_panorama.py`)
- pose authority는 FAST-LIO2 `/Odometry`
- Habitat API 대체는 Phase 1 산출물 기준으로 수행

## 2) 기준 문서/코드

기준 문서:
- `omninav_realworld_plan.md`
- `docs/realworld/habitat_replacement_map.md`
- `docs/realworld/topic_io_contract_v1.md`
- `docs/realworld/habitat_deps_raw.txt`

핵심 코드(재사용/참조):
- slow 오리지널 루프: `infer_ovon_slowfast/run_nav_ovon_omni.py`
- slow decision 핵심: `infer_ovon_slowfast/qwen_utils.py:getresult`
- frontier 로직: `infer_ovon_slowfast/frontier_utils.py`
- fast 실행체: `infer_r2r_rxr/run_infer_online_panorama.py`
- 액션 실행: `infer_r2r_rxr/omninav_control.py`
- OGMap 참고: `scripts/gridmapper.cpp`

## 3) OVON 원래 로직에서 유지할 부분

유지 대상:
- frontier 후보 집합 생성 -> decision model -> `target_position(subgoal)` 선택 흐름
- `getresult(...)`가 frontier 좌표를 보고 목적 좌표를 고르는 패턴
- visited frontier 관리, 재계획 루프 구조

실세계 치환 대상:
- `maps.get_topdown_map_from_sim` -> `/omni/occupancy`
- `agent.get_state` -> `/Odometry`(또는 `/omni/pose2d`)
- `sim.get_sensor_observations` -> `/cam_front|left|right/color/image_raw`
- `pathfinder.*`/`GreedyGeodesicFollower` -> occupancy-grid 기반 경로/가용성 체크

## 4) Phase 3 작업 패키지

## WP-A: `lio_grid_node` (Python/ROS2 v1, C++ v2 선택) 구현

목적:
- FAST-LIO2 포인트클라우드와 pose로 slow 입력용 2D OGMap 생성

입력/출력:
- 입력: `/cloud_registered` (`sensor_msgs/PointCloud2`), `/Odometry` (`nav_msgs/Odometry`)
- 출력: `/omni/occupancy` (`nav_msgs/OccupancyGrid`), `/omni/pose2d` (`geometry_msgs/PoseStamped`)

`scripts/gridmapper.cpp` 재사용 포인트:
- 재사용 가능:
  - Bresenham ray-casting (`scripts/gridmapper.cpp:95`)
  - log-odds 업데이트 (`scripts/gridmapper.cpp:34`)
  - 맵 자동 확장 grow (`scripts/gridmapper.cpp:46`)
- 변경 필요:
  - ROS1 API -> ROS2 `rclcpp` (`scripts/gridmapper.cpp:1`, `scripts/gridmapper.cpp:231`)
  - `mm::Frame` 구독 제거, `PointCloud2 + Odometry` 동기화 구독으로 교체 (`scripts/gridmapper.cpp:137`, `scripts/gridmapper.cpp:178`)
  - `height slice` 추가 (`cloud_registered`는 3D 전체 포인트)
  - frame_id/TF 체인 검증 (`map` 기준 발행 유지, 현재 체인과 정합)

구현 상세(초안):
- v1 구현 언어: Python(`rclpy`) 우선
- 동기화: `message_filters` ApproximateTime (`PointCloud2`, `Odometry`)
- 높이 필터:
  - 기본안: `z_rel = point.z - odom.pose.pose.position.z`
  - 사용 범위 파라미터: `slice_z_min`, `slice_z_max` (예: `0.10 ~ 0.50 m`)
- log-odds:
  - free update: `-0.4`, occupied update: `+0.85` (참조 코드 동일)
- occupancy 변환:
  - unknown/free/occupied 3상태 유지 (`-1/0~49/50~100`)
- publish rate: `1~2 Hz` (slow 주기 맞춤)
- v2 최적화(선택): CPU 여유가 부족하면 C++로 이식

완료 기준:
- `/omni/occupancy` 안정 발행 (>=1Hz, 목표 1~2Hz)
- RViz에서 map 프레임 기준으로 cloud/pose와 정합
- frontier 추출 가능한 맵 품질 확보

## WP-B: Memory Bank 경계 (v1은 adapter 내부 모듈)

목적:
- frontier 대표 이미지 조회를 위한 tri-view + pose 히스토리 버퍼 제공
- 초기 데모 복잡도/통신 오버헤드를 줄이기 위해 v1은 `habitat_adapter_node` 내부 모듈로 포함

입력/출력:
- 입력: `/cam_front|left|right/color/image_raw`, `/omni/pose2d`(또는 `/Odometry`)
- 출력(v1): adapter 내부 함수 호출
- 출력(v2): 필요 시 `/omni/frontier_views` 서비스로 분리

구현 상세(초안):
- 동기화: 3카메라 + pose ApproximateTime
- 저장 구조: ring buffer (`timestamp`, `pose`, `front/left/right`)
- 조회 규칙:
  - subgoal/frontier 좌표에 가장 가까운 pose 샘플 반환
  - 시간 오차, 거리 오차 임계치 파라미터화

완료 기준:
- N개 frontier 질의 시 N개 tri-view 일관 반환
- 10분 연속 동작 중 메모리 누수/버퍼 폭주 없음

## WP-C: `habitat_adapter_node` (Python/ROS2) 구현

목적:
- Habitat 의존 slow 루프를 ROS 입력 기반으로 치환
- `/omni/subgoal` 생성까지 담당

입력/출력:
- 입력: `/omni/occupancy`, `/omni/pose2d`, memory bank 조회 결과
- 출력: `/omni/frontiers`, `/omni/subgoal`

로직 구성:
1. occupancy에서 frontier 후보 추출 (`frontier_utils` 재사용 중심)
2. frontier 후보를 `getresult(...)` 입력 형식으로 변환
3. selected `target_position`을 `/omni/subgoal`로 publish
4. visited frontier set / stale timeout 관리

운영 모드 원칙:
- 성능 테스트/데모는 `decision_mode=model` 사용
- `rule` 모드는 bring-up 또는 모델 백엔드 실패 시 fallback 용도로만 유지
- 모델 경로는 컨테이너 기준 `/workspace/OmniNav/OmniNav_Slowfast` 고정
- 런타임 충돌 방지를 위해 adapter 실행 시 UCX `LD_PRELOAD` 가드(기존 fast 스크립트와 동일 정책) 적용

`getresult(...)` 입력 계약 (Phase 3에서 고정):
- `current_frontiers`: `List[np.ndarray]`, 각 원소 shape `(3,)`, 좌표계 `map`, 단위 `m`
- `decision_agent_state.position`: `np.ndarray([x,y,z])`, 좌표계 `map`
- `decision_agent_state.rotation`: 필드 접근형(`.x/.y/.z/.w`) 쿼터니언
- `bank.get_spin_data()`: OVON 원본과 동일하게 360 spin 샘플 + 기준 뷰를 포함한 이미지 리스트 제공
- `visited_frontier_set`: `set(tuple(round(x,1), round(y,1), round(z,1)))` 형식 유지

주의:
- adapter는 `/action` 발행 금지
- `pathfinder` 대체는 occupancy-grid 가용성 검사 + A* 기반으로 구현
- 구현 상세(A* cost/snap/fallback)는 본 Phase에서 확정
- bank 기능은 v1에서 adapter 내부 모듈로 구현하고, 성능/재사용 요구가 생기면 노드/서비스로 분리

완료 기준:
- slow decision 주기 1~2Hz 유지
- frontier 선택 -> subgoal publish 연속 동작

## WP-D: fast 연동 (최소 변경)

목적:
- fast 추론체는 유지하면서 실세계 입력만 교체

필수 변경:
- `run_infer_online_panorama.py`의 고정 pose 제거
  - 현재: `default_pose` 사용 (`infer_r2r_rxr/run_infer_online_panorama.py:432`)
  - 목표: `/Odometry`(또는 `/omni/pose2d`) 실 pose 주입

pose 좌표계 검증/변환(필수):
- `Waypoint_Agent.pose_to_matrix()`가 기대하는 형식에 맞춰 회전/축 정의를 명시적으로 맞춘다.
- 주입 규약:
  - `position`: `[x, y, z]` (map frame, meter)
  - `rotation`: `[w, x, y, z]` (기존 코드 기대 형식 유지)
- 사전 검증:
  - 로봇 전진 시 local pose `z/x` 부호 일관성
  - 좌회전/우회전 시 heading 변화 방향 일관성
  - 1m 직진 테스트에서 누적 pose 오차 허용범위 내인지 확인

subgoal 연동 원칙:
- OVON 원래 흐름처럼 slow에서 결정한 subgoal이 fast 단계 입력으로 전달되는 데이터 경로를 유지
- 단, 모델 내부 아키텍처는 변경하지 않고 입력 어댑팅 레이어에서 처리

완료 기준:
- fast가 실 pose 기반으로 `/action` 발행
- `/action` 발행자는 기존 단일 노드 유지

## WP-D2: Coordinate Token Fusion (원본 fast 계약 정합)

목적:
- slow가 만든 subgoal을 텍스트 힌트가 아닌 좌표 토큰으로 fast 모델에 주입한다.
- 모델/가중치 변경 없이 기존 `action_former=True` 경로를 그대로 사용한다.

계약(원본 `infer_ovon_slowfast/qwen_utils.py` 정합):
- token 순서: `hist4 -> current -> target(subgoal)`
- token 개수: `N=6`
- shape: `[1, 6, 2]` (모델 입력 시 배치 차원 포함)
- 스케일: `input_waypoints_scaled = input_waypoints / 0.3`

좌표계/변환:
- ROS map `(x,y)`는 Habitat-like `(x,0,y)`로 변환하여 사용
- subgoal local 변환:
  - `p_world = [subgoal_x, 0, subgoal_y, 1]`
  - `p_local = inv(T_current) @ p_world`
  - token에는 `(p_local_x, p_local_z)` 사용
- `T_current`는 `Waypoint_Agent.pose_to_matrix()` 경로를 재사용해 축 정의 일관성을 유지

결측/예외 정책:
- subgoal 없음 또는 변환 실패 시 `input_waypoints=None`으로 fallback
- 모델 forward가 `input_waypoints`를 지원하지 않으면 키를 제거하고 재시도

실험 플래그:
- `run_infer_online_panorama.py --use-coordinate-tokens`
- 좌표 토큰 단독 효과 검증 시 `--no-subgoal-hint` 병행

## 5) 구현 순서(권장)

1. WP-A `lio_grid_node` Python v1  
2. WP-C `habitat_adapter_node` + 내부 memory bank(v1)  
3. WP-D fast pose/subgoal 연동  
4. 통합 검증  
5. 필요 시 WP-A C++ 최적화 + WP-B 서비스 분리(v2)

이 순서를 권장하는 이유:
- slow 입력 3요소(map/pose/view)가 먼저 안정화되어야 adapter 품질 검증이 가능함

## 6) 통합 검증 시나리오

시나리오 S1 (노드 단위):
- `/cloud_registered` + `/Odometry` -> `/omni/occupancy` 생성 확인
- tri-view + pose -> memory bank 조회 확인

시나리오 S2 (slow 단위):
- occupancy 입력에서 frontier 추출
- `getresult` 호출 후 `/omni/subgoal` 발행 확인

시나리오 S3 (폐루프):
- `/omni/subgoal` 업데이트 중 fast 추론 지속
- `/action` -> `omninav_control` -> `cmd_vel` 연동
- 10분 런에서 루프 중단/토픽 끊김 없음

## 7) Gate G3 (Phase 3 완료 기준)

- `/omni/occupancy` 1~2Hz 안정 출력
- `/omni/frontiers`, `/omni/subgoal` 안정 출력
- fast 입력 pose가 실측값으로 교체됨
- `/action` 단일 발행자 원칙 유지
- 실제 로봇에서 slow->fast->control 폐루프 10분 연속 동작

## 8) 리스크 및 대응

R1. 카메라 저FPS/지연  
- 대응: memory bank 조회에서 timestamp/거리 임계치, 최근접 fallback

R2. occupancy 품질 불안정  
- 대응: height slice 파라미터 튜닝, inflation/ray-casting gain 조정

R3. subgoal과 fast 제어 불일치  
- 대응: subgoal stale timeout + 재계획 주기 고정(1~2Hz)

R4. TF/frame 불일치 재발  
- 대응: 런치 시 `verify_tf.py`를 preflight 체크로 고정

## 9) 산출물

- 구현 코드:
  - `lio_grid_node` (ROS2 package)
  - `rgb_memory_bank_node` (ROS2 package)
  - `habitat_adapter_node` (ROS2 package)
  - fast pose 입력 연동 패치
- 문서:
  - 본 문서(`docs/realworld/phase3_execution_plan_v1.md`)
  - 런치 순서/검증 명령 정리 문서(별도)

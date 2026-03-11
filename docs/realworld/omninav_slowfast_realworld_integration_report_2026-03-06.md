# OmniNav Slow+Fast 실세계 적용 통합 보고서

작성일: 2026-03-06  
작성 범위: `OmniNav_setup` 리포지토리 기준 실구현/실행 경로

---

## 0. 보고서 목적

기존 OVON 데이터셋/Habitat 기반 OmniNav slow+fast 파이프라인을, FAST-LIO2 + Scout Mini + Orbbec 3Cam 실세계 스택으로 치환한 내용과 현재 추론/제어 동작 방식을 코드 기준으로 정리한다.  
또한 현재 상태에서 성능/안정성 상 발생 가능한 문제를 타당한 원인과 함께 정리한다.

---

## 1. 기준 문서 및 코드

### 1-1. 계획/정리 문서

- `omninav_realworld_plan.md`
- `docs/realworld/habitat_replacement_map.md`
- `docs/realworld/topic_io_contract_v1.md`
- `docs/realworld/phase3_execution_plan_v1.md`
- `docs/realworld/weekly_report_tf_focus_2026-02-26.md`

### 1-2. 핵심 구현 코드

- 원본 slow+fast 참조
  - `infer_ovon_slowfast/run_nav_ovon_omni.py`
  - `infer_ovon_slowfast/qwen_utils.py`
  - `infer_ovon_slowfast/frontier_utils.py`
- 실세계 적용 코드
  - `scout_tf_config/scripts/lio_grid_node.py`
  - `scout_tf_config/scripts/habitat_adapter_node.py`
  - `infer_r2r_rxr/run_infer_online_panorama.py`
  - `infer_r2r_rxr/agent/waypoint_agent.py`
  - `infer_r2r_rxr/omninav_control.py`
  - `scout_tf_config/launch/sensor_tf_launch.py`

---

## 2. 원본 OVON(Habitat) 파이프라인 요약

원본은 Habitat simulator API를 직접 호출하는 구조다.

- slow:
  - `maps.get_topdown_map_from_sim(...)`로 지도 취득
  - `sim.get_sensor_observations()`, `agent.get_state()`로 관측/포즈 취득
  - `getresult(...)`로 frontier 중 target(subgoal) 결정
- fast:
  - `get_result_fast(...)` 경로에서 waypoint 예측
- 제어/이동:
  - `sim.step(...)`, `pathfinder.*`, `GreedyGeodesicFollower`로 시뮬레이터 내 이동

즉, 핵심 정책 로직은 모델 기반이나 입출력 인터페이스는 Habitat 의존도가 높았다.

---

## 3. 실세계 적용 목표와 고정 원칙

### 3-1. 목표

- slow는 실측 Occupancy + frontier 기반 subgoal 선택
- fast는 실카메라 3뷰 + 실포즈로 waypoint 추론
- control은 `/action -> /scout_mini_base_controller/cmd_vel` 폐루프

### 3-2. 고정 원칙

- 모델 가중치/핵심 추론 구조는 유지
- Habitat API는 ROS2 토픽/노드로 치환
- `/action` 발행자는 fast 스크립트 단일 유지
- pose authority는 FAST-LIO2 계열(`/Odometry` -> `/omni/pose2d`)로 통일

---

## 4. 무엇을 어떻게 바꿨는지 (구현 상세)

## 4-1. Habitat 의존 치환 설계 정리

문제:
- 원본 스크립트가 `sim.*`, `maps.*`, `pathfinder.*`에 강하게 결합되어 실로봇 실행 불가.

조치:
- `habitat_replacement_map.md`에 API별 대체원 정의.
- `topic_io_contract_v1.md`에 실제 토픽 계약 확정.

결과:
- "대체 대상"과 "ROS 입력원"이 분리 정리되어 구현 경계가 명확해짐.

## 4-2. TF/좌표계 기반 정리 (Phase 2)

목적:
- FAST-LIO 포즈, LiDAR, 3카메라를 동일 체인으로 정렬.

조치:
- `sensor_tf_launch.py`로 정적 TF + odom TF 브리지를 일괄 실행.
- `sensor_tf.yaml`에 실측/보정값 반영.
- `verify_tf.py`를 통한 체인 검증 경로 유지.

핵심 역할:
- `map -> base_link -> lidar_frame -> cam_*_optical_frame` 체인 일관성 확보.

## 4-3. `lio_grid_node` 추가 (WP-A)

목적:
- FAST-LIO 포인트클라우드를 slow 입력용 2D occupancy로 변환.

입력:
- `/cloud_registered` (PointCloud2)
- `/Odometry` (Odometry)

출력:
- `/omni/occupancy`
- `/omni/pose2d`

주요 구현:
- `ApproximateTimeSynchronizer`로 cloud/odom 동기화
- height slice(`slice_z_min/max`) 적용
- ray-casting + log-odds 업데이트 + map grow + inflation
- publish rate 2Hz
- `max_points_per_scan`을 8000으로 제한해 CPU 부하 완화

의미:
- Habitat topdown map 대체를 실제 LIO map으로 수행.

## 4-4. `habitat_adapter_node` 구현 (WP-C, memory bank 내장)

목적:
- 실세계 occupancy/pose/image를 받아 slow subgoal 생성.

입력:
- `/omni/occupancy`, `/omni/pose2d`, `/cam_front|left|right/color/image_raw`

출력:
- `/omni/frontiers`, `/omni/subgoal`, `/omni/instruction`

주요 구현:
- occupancy free-unknown 경계 기반 frontier 추출
- 3카메라+pose 동기화로 internal memory bank 구성
- `decision_mode=model`에서 OVON형 `getresult` 호환 경로 사용
- 모델 백엔드 미준비 시 `rule` fallback 유지
- `decision_hz` 주기로 subgoal 갱신, stale timeout/visited 관리

중요 차이:
- 원본 frontier는 Habitat map/fog-of-war 기반.
- 현재는 실 occupancy(grid) 경계 기반으로 구현.

## 4-5. `run_infer_online_panorama.py` 실세계 fast 추론 경로로 확장

목적:
- fast를 실세계 입력으로 동작시키고 slow subgoal을 조건으로 주입.

주요 변경:
- tri-view를 `message_filters` 동기화로 취득
- sync gate 도입
  - `max_camera_skew_ms` (기본 200)
  - `max_pose_dt_ms` (기본 150)
  - `max_subgoal_age_ms` (기본 3000)
- `/omni/pose2d`를 모델 pose로 변환해 사용
- `/omni/subgoal`을 coordinate token 경로로 전달
- fallback/정합 로그 강화
  - `[FALLBACK]`, `[ALIGN]` 출력
- 결과 저장 강화
  - run 로그 자동 저장(`results/run_online_*.log`)
  - CSV 저장(`waypoint_data_online_*.csv`)
  - MP4 저장(`omninav_online_*.mp4`)
  - vis frame ring buffer(`max_vis_frames`) 적용

기본 실행 성격:
- 현재 기본은 coordinate token 활성, subgoal text hint 비활성.

## 4-6. `waypoint_agent.py` 추론/리소스 안정화 패치

주요 조치:
- coordinate token 경로 추가/유지
  - `hist4 + current + target` (N=6) 구성
  - target은 subgoal world -> local 변환 사용
- 논문형 이미지 스케일 반영
  - current tri-view: `480x426`
  - history buffer: `120x106`, 최대 20프레임
- 디버그 PNG 상시 저장 비활성화
- per-frame `torch.cuda.empty_cache()` 제거 (할당/해제 부하 억제)
- fallback 카운트/사유 집계

## 4-7. `omninav_control.py` 제어 방식

입력:
- `/action` (JSON: 5개 waypoint + arrive)

출력:
- `/scout_mini_base_controller/cmd_vel` (TwistStamped)

제어 로직:
- 1개의 `/action` 배치에 포함된 5개 waypoint를 순차 실행
- waypoint당 `0.15s` 실행 (총 약 `0.75s`)
- 곡률 기반 추종(curvature pursuit)으로 `(dx,dy)->(v,w)` 변환
- `arrive_pred>0`이면 즉시 정지

---

## 5. 현재 실제 추론/제어 데이터 플로우

1. 센서 계층
- Livox + FAST-LIO2: `/cloud_registered`, `/Odometry`
- 3카메라: `/cam_front|left|right/color/image_raw`

2. 맵/포즈 계층
- `lio_grid_node`가 `/omni/occupancy`, `/omni/pose2d` 발행

3. slow 계층
- `habitat_adapter_node`가 occupancy에서 frontier 추출
- memory bank + instruction 기반으로 subgoal 선택
- `/omni/subgoal` + `/omni/instruction` 발행

4. fast 계층
- `run_infer_online_panorama.py`가 tri-view + pose + subgoal 동기 검증
- Qwen 기반 waypoint 추론 후 `/action` 발행

5. control 계층
- `omninav_control.py`가 `/action`을 받아 cmd_vel로 순차 실행

즉, 현재는 "slow subgoal 생성"과 "fast waypoint 생성"이 분리 프로세스로 유지되고, 제어는 배치 단위 순차 실행 구조다.

---

## 6. 현재 동작 특성 (실행 로그 기준)

### 6-1. 추론 속도

- fast 추론 1프레임에 대략 5~6초 소요(로그 기준)
- 결과적으로 실효 FPS는 약 0.18 수준

### 6-2. 제어 타이밍

- 제어 자체는 10Hz 루프이나, 상위 `/action` 갱신이 추론 완료 시점에 종속
- 고비용 추론 프레임 동안은 새로운 행동 배치 갱신 주기가 길어짐

### 6-3. 동기화 게이트 동작

- 카메라 skew/pose dt/subgoal age 조건 미충족 시 프레임 드롭
- 이는 잘못된 프레임 혼합을 막는 대신 유효 추론 샘플 수를 감소시킴

---

## 7. 현재 상태에서 발생 가능한 문제와 타당한 이유

## 7-1. 체감 주행 속도 저하

현상:
- "추론하고 제어를 하나씩 하는" 느낌으로 느림.

이유:
- fast 1회 추론 시간이 길어 `/action` 갱신 주기가 느림.
- control은 마지막 배치(5 waypoint)를 실행한 뒤 새 배치를 기다리므로 상위 추론이 느리면 전체 반응이 느려짐.

## 7-2. 프레임 드롭 증가 시 반응성 저하

현상:
- `[SYNC] drop frame` 로그가 많이 뜨면 의사결정 간격이 늘어남.

이유:
- 현재는 동기 신뢰성 우선 정책이므로 gate 기준을 넘는 샘플은 폐기.
- 카메라 timestamp skew가 클수록 추론 입력 채택률이 낮아짐.

## 7-3. subgoal 방향과 wp0 방향 불일치 가능

현상:
- `[ALIGN] angle_err`가 큰 프레임 존재.

이유:
- coordinate token은 "조건"이며 최종 waypoint는 이미지/텍스트와 융합 결과.
- 로컬 좌표계/뷰 컨텍스트/장애물 회피 성향이 결합되면 즉시 wp0가 subgoal 직선방향과 다를 수 있음.

## 7-4. `/action` 소비자 일시 미연결 문제

현상:
- `No subscribers for /action` 경고 프레임 발생 가능.

이유:
- control 노드 재시작/중단 또는 ROS graph 연결 지연 시 발행 시점 미스매치 발생.

## 7-5. 시스템 부하로 인한 불안정(멈춤/재부팅 체감)

현상:
- 장시간 구동 중 시스템 먹통/비정상 종료 가능성 제기.

이유(복합):
- slow 모델 + fast 모델 동시 로드(2회 로드) + 센서/SLAM 동시 구동으로 CPU/GPU/메모리/IO 부하 중첩.
- MP4/CSV/런타임 로그 기록과 실시간 추론이 동시에 돌아가 IO 경합 가능.
- 커널/드라이버 조합 불안정 시 부하 상황에서 시스템 레벨 오류 확률 증가.

---

## 8. 현재 상태 결론

- 목표했던 "실세계 slow+fast 데모 가능" 상태는 달성됨.
- 원본 OVON의 핵심 흐름(frontier -> subgoal -> waypoint -> control)은 유지하면서 Habitat I/O를 ROS2 실세계 입력으로 치환 완료.
- 다만 실사용 품질 관점에서는 다음 3가지가 주요 병목이다.
  - fast 단일 프레임 추론 지연(수 초 단위)
  - 엄격한 동기화 게이트로 인한 입력 드롭
  - 추론-제어 업데이트 주기 비대칭으로 인한 체감 지연

즉, 현재는 "기능 통합/재현 가능 단계"이며, "속도/안정성 최적화 단계"가 다음 우선 과제다.


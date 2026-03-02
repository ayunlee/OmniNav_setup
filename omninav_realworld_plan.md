# OmniNav Slow+Fast 실세계 데모 실행 계획서 (Scout Mini + Livox Mid-360 + Orbbec 3Cam)

작성일: 2026-02-23  
대상 리포지토리: `/home/aprl_msi/ayun/OmniNav_setup`

---

## 1) 목표

FAST-LIO2(ROS2) 출력을 OmniNav slow+fast 시스템의 실세계 입력으로 연결해, Scout Mini에서 폐루프 데모를 수행한다.

데모 성공 기준:
- `slow`: occupancy + frontier 기반으로 subgoal 선택
- `fast`: 3방향 RGB + pose 기반 waypoint 추론
- `control`: `/action -> /scout_mini_base_controller/cmd_vel` 연동

---

## 2) 비기능 제약 (고정 원칙)

### 2.1 변경 금지
- OmniNav 모델 추론 로직(가중치, forward 흐름, 토치/트랜스포머 핵심 버전) 변경 금지
- 성능에 영향 큰 내부 로직 변경 금지

### 2.2 허용 범위
- ROS 브리지/어댑터/맵 변환 노드 추가
- 입력 소스 연결(토픽/포즈 주입/좌표 변환)
- 실행 스크립트 및 런치 구성 정리

### 2.3 최소 신규 컴포넌트
- `lio_grid_node` (필수)
- `rgb_memory_bank_node` (필수)
- `habitat_adapter_node` (핵심)
- 기존 재사용: `infer_r2r_rxr/omninav_control.py`

### 2.4 최종 의사결정 (확정)
- Q1 Fast 실행 주체: `infer_r2r_rxr/run_infer_online_panorama.py` 유지
- Q2 Pathfinder 대체: occupancy-grid 기반 A* 채택
- Q3 TF/Pose authority: FAST-LIO2 (`/Odometry`)를 단일 기준으로 사용

### 2.5 토픽 발행 권한 원칙
- `/action` 단일 발행자: `run_infer_online_panorama.py`만 발행
- `habitat_adapter_node`는 `/action`을 발행하지 않음
- adapter 출력은 `/omni/subgoal`(및 디버그 토픽)으로 제한

Phase 경계 원칙:
- `run_infer_online_panorama.py`의 실 pose 주입은 **Phase 3 범위**이며, Phase 2에서는 TF/동기화만 진행

---

## 3) 현재 코드 기준 재사용 포인트

### 3.1 Fast/Control 재사용 지점
- `infer_r2r_rxr/run_infer_online_panorama.py:219`  
  `/cam_front|left|right/color/image_raw` 구독
- `infer_r2r_rxr/run_infer_online_panorama.py:239`  
  `/action` 퍼블리시
- `infer_r2r_rxr/run_infer_online_panorama.py:432`  
  현재 `default_pose` 고정값 사용 (실세계 pose 연동 필요)
- `infer_r2r_rxr/omninav_control.py:60`  
  `/action` 구독
- `infer_r2r_rxr/omninav_control.py:69`  
  `/scout_mini_base_controller/cmd_vel` 퍼블리시

### 3.2 Slow 재사용 지점
- `infer_ovon_slowfast/run_nav_ovon_omni.py:96`  
  `maps.get_topdown_map_from_sim` (Habitat 의존)
- `infer_ovon_slowfast/run_nav_ovon_omni.py:125`  
  `sim.step` (Habitat 의존)
- `infer_ovon_slowfast/run_nav_ovon_omni.py:178`  
  `GreedyGeodesicFollower` (Habitat 의존)
- `infer_ovon_slowfast/qwen_utils.py:96`  
  `getresult(...)` (slow decision 핵심, 재사용 대상)
- `infer_ovon_slowfast/frontier_utils.py:459`  
  `detect_frontiers(...)` (frontier 알고리즘 재사용 가능)

---

## 4) 전체 실행 전략 (병렬 + 게이트)

- Phase 1: 분석 + 하드웨어 bring-up 병렬
- Phase 2: TF/시간동기 독립 phase
- Phase 3: 브리지 3노드 개발
- Phase 4: 통합/데모 런북 고정

각 phase는 Gate 통과 후 다음 단계로 진행한다.

---

## 5) Phase 0 (30분) - 최소 베이스라인 캡처

목적: 문서작업 최소화하면서 재현성만 확보

작업:
- 현재 동작 중인 OmniNav 추론 환경 버전 캡처
  - python/torch/transformers/cuda
  - 모델 경로
  - git commit hash

완료 산출물:
- `docs/realworld/baseline_env.txt` (간단 텍스트 1개)

주의:
- 이 단계는 "문서 작업"이 아니라 롤백 대비 최소 스냅샷이다.

---

## 6) Phase 1 (1주차) - 분석 + 하드웨어 병렬

## Track A: Habitat 의존성 전수 분석 + 치환표

### A-1. 의존성 스캔
실행 예시:
```bash
cd /home/aprl_msi/ayun/OmniNav_setup
rg -n "sim\\.|habitat|pathfinder|maps\\." infer_ovon_slowfast infer_r2r_rxr --glob "*.py"
```

주의:
- A-1에서 `sim.step()`, `sim.reset()`, `sim.geodesic_distance()` 등 예상보다 많은 의존성이 나올 수 있음
- 아래 단계(A-2)는 A-1 실측 결과를 기준으로 채움 (예상 목록 고정 금지)

### A-2. 치환 매핑표 작성 (A-1 결과 기반)
형식:
- `old_api -> replacement_required -> candidate_ros_source`

초기 예시(확정 목록 아님):
- `sim.get_agent_state()` -> 대체 필요 -> `/Odometry` 또는 `/tf`
- `maps.get_topdown_map_from_sim(...)` -> 대체 필요 -> `/omni/occupancy`
- `sim.get_sensor_observations()` -> 대체 필요 -> `/cam_*` RGB 토픽
- `pathfinder.*` -> 대체 필요 -> occupancy grid 기반 경로탐색

범위 원칙:
- A-2는 "무엇을 대체할지"까지만 확정
- 구현 상세(A* 방식, snap_point 로직, fallback 정책)는 Phase 3(`habitat_adapter_node`)에서 결정

### A-3. 입력/출력 스펙 1차 확정
이 단계에서는 1차 스펙만 확정:
- slow 입력 타입 (1차)
- fast 입력 타입 (1차)
- subgoal handoff 타입 (1차)
- 좌표계 정의(`map`, `odom`, `base_link`) (1차)

운영 원칙:
- 1차 확정 후 Phase 3 통합 시 필요하면 수정 가능

Track A 완료 기준:
- `docs/realworld/habitat_replacement_map.md` 완성
- `docs/realworld/habitat_deps_raw.txt` 작성
- `docs/realworld/topic_io_contract_v1.md` 완성

---

## Track B: 하드웨어 + FAST-LIO2 bring-up

### B-1. 물리 구성
- Livox Mid-360 Scout Mini 상판 고정
- 전원/이더넷 연결
- Orbbec 3대 장착 방향(Front/Left/Right) 확정

### B-2. Scout/Orbbec 재확인
- Scout base controller + teleop 확인
- `triple_camera_launch.py` 기준 3카메라 토픽 수신 확인

### B-3. Livox + FAST-LIO2 설치
- `livox_ros_driver2` 설치/빌드
- FAST-LIO2 ROS2 브랜치 클론/빌드
- Mid-360 config 적용

호환성 체크(필수):
- FAST-LIO 빌드 의존 패키지명 확인: `livox_ros2_driver`
- 실제 설치 패키지명 확인: `ros2 pkg list | rg livox`
- 불일치 시 launch/토픽명 수정 전에 패키지명 호환부터 해결

초기 데모 파라미터 기준:
- `FAST_LIO2/FAST_LIO_SLAM_ros2/config/mid360.yaml`
- `publish.path_en: true`, `publish.scan_publish_en: true`
- `pcd_save.pcd_save_en: false` (장시간 데모 메모리/디스크 보호)
- `mapping.extrinsic_est_en`: 외부 보정치가 확정이면 `false`, 아니면 `true`로 시작 후 고정

### B-4. 기본 시각화 확인
- RViz에서 cloud/odom/path 확인

Track B 완료 기준:
- `/cloud_registered` 수신
- `/Odometry` 수신
- `/path` 수신
- 3방향 RGB 토픽 수신
- FAST-LIO frame 확인: odom frame=`camera_init`, child=`body` 파악 완료

---

## Gate G1 (Phase 1 종료 조건)

다음 5개가 동시에 만족:
- Habitat 치환표 완료
- 토픽 I/O 스펙 확정
- FAST-LIO2 토픽 안정 출력 (`/Odometry >= 8Hz`, `/cloud_registered >= 8Hz`)
- 3카메라 안정 출력 (각 카메라 토픽 `>= 8Hz`)
- Scout 수동 주행 가능

---

## 7) Phase 2 (충분한 시간 확보) - TF/시간동기

이 단계는 실제로 며칠 소요 가능한 핵심 난이도 작업이다.

### 2-1. TF 체인 정리
목표 체인:
- `map -> base_link -> lidar_frame -> cam_front/left/right_color_optical_frame`

작업:
- FAST-LIO native frame 매핑 정의 (`camera_init -> map`, `body -> base_link`)
- Scout wheel odom TF 비활성화 (`scout_mini.yaml: enable_odom_tf=false`)
- 정적 extrinsic 반영 (`base_link <-> lidar_frame`, `base_link <-> camera_*`)
- `base_link -> lidar_frame` 초기값은 실측 환산값 사용 (`x=0.09, y=0.0, z=0.242`)
- 프레임 네이밍 표준화
- `ros2 run tf2_tools view_frames`로 구조 확인

### 2-2. 타임스탬프 일관성 검증
작업:
- `/Odometry`, `/cloud_registered`, `/cam_*` timestamp 오프셋 확인
- 지연/드롭 체크(`ros2 topic hz`, `ros2 topic delay` 대체 스크립트)

허용 기준:
- RGB-POSE 매칭 오차 `<= 100 ms`
- 10분 연속 구동 중 누적 drop으로 의사결정 루프 중단 없음

### 2-3. RViz 정합 검증
검증:
- 로봇 이동 시 cloud와 pose가 정상 추종
- RGB 관측 방향과 로봇 yaw 일치
- 프레임 드리프트/축 뒤집힘 없음

완료 기준 (Gate G2):
- RViz에서 cloud + odom + RGB 프레임 정합
- 장시간(>=10분) 구동 시 프레임 붕괴 없음
- TF 루프/중복 authority 없음 (frame publisher 단일화 확인)

---

## 8) Phase 3 - 브리지 3노드 개발 (실제 FAST-LIO2 데이터 기반)

## 3-1. `lio_grid_node` (필수)

입력:
- `/cloud_registered` (`sensor_msgs/PointCloud2`)
- `/Odometry` (`nav_msgs/Odometry`) 또는 TF

출력:
- `/omni/occupancy` (`nav_msgs/OccupancyGrid`)
- `/omni/pose2d` (`geometry_msgs/PoseStamped` 또는 전용 토픽)

요구사항:
- height slice + ray-casting 포함
- 맵 상태 3값 유지: unknown / free / occupied
- slow 루프용 1~2Hz 안정 publish

로봇 안전 규칙:
- inflation 적용 (Scout footprint + safety margin)
- 기본 정책: `unknown`은 통과 불가

검증:
- frontier 후보가 안정적으로 검출될 정도의 맵 품질

---

## 3-2. `rgb_memory_bank_node` (필수)

입력:
- `/cam_front/color/image_raw`
- `/cam_left/color/image_raw`
- `/cam_right/color/image_raw`
- `/omni/pose2d` 또는 `/Odometry`

역할:
- (pose, tri-RGB, timestamp) ring buffer 저장
- frontier 좌표 질의 시 최근접 관측 반환

출력:
- adapter가 frontier별 대표 이미지 조회 가능한 API(서비스/내부 호출)

검증:
- frontier N개 입력 시 N개 이미지 안정 반환

---

## 3-3. `habitat_adapter_node` (핵심, 최대 작업량)

역할:
- Habitat API 호출부를 ROS 입력으로 대체
- slow decision 경로(`getresult`) 재사용
- 선택된 subgoal을 fast 입력으로 전달 (`/omni/subgoal`)

원칙:
- 모델 로직은 변경하지 않고 입력 소스만 교체
- `infer_ovon_slowfast/qwen_utils.py:96` slow 의사결정 로직 재사용

중요:
- fast 실세계에서는 pose 고정값 금지
- `infer_r2r_rxr/run_infer_online_panorama.py:432`의 `default_pose`는 실 pose 입력으로 대체 필요  
  (모델 구조 변경이 아니라 입력 데이터 연결 작업)

추가 요구사항:
- `/action` 발행 금지 (단일 발행자 원칙 유지)
- subgoal stale timeout 적용 (예: 2s 초과 시 재계획)

Phase 3 완료 기준 (Gate G3):
- slow가 frontier 선택
- fast가 실 pose를 입력으로 사용
- fast에 subgoal 전달
- `/action` 유효 메시지 생성 (발행자 1개만 존재)

---

## 9) Phase 4 - 통합/데모

## 4-1. 폐루프 통합
데이터 플로우:
`adapter -> /omni/subgoal -> run_infer_online_panorama -> /action -> omninav_control -> /scout_mini_base_controller/cmd_vel`

검증:
- waypoint 수신/실행 정상
- arrive_pred 시 정지 정상
- 명령 누락/폭주 없음
- stale `/action` timeout 시 자동 정지

## 4-2. 데모 런북 고정
런북 포함 항목:
- 실행 순서
- 필수 토픽 health check
- 장애 복구 순서(카메라/SLAM/컨트롤러 재시작)
- 안전 정지 절차

완료 기준:
- 동일 절차로 3회 연속 재현 성공

---

## 10) 작업 우선순위 요약

1. Track A/B 병렬 시작 (분석 + Livox/FAST-LIO2 bring-up)  
2. TF/동기화 선해결 (Gate G2 통과 전 브리지 개발 최소화)  
3. 브리지 3노드 개발  
4. 폐루프 통합 + 런북 고정

---

## 10-1) 즉시 실행 체크리스트 (명령어 템플릿)

### A. Habitat 분석 트랙
```bash
cd /home/aprl_msi/ayun/OmniNav_setup
rg -n "sim\\.|habitat|pathfinder|maps\\." infer_ovon_slowfast infer_r2r_rxr --glob "*.py" > /tmp/habitat_deps.txt
```
산출물:
- `/tmp/habitat_deps.txt`
- `docs/realworld/habitat_replacement_map.md`

### B. 하드웨어/SLAM 트랙
```bash
# 토픽 존재 확인
ros2 topic list | rg "cloud_registered|Odometry|path|cam_front|cam_left|cam_right"

# livox 패키지명 확인 (의존성 호환)
ros2 pkg list | rg "livox_ros2_driver|livox_ros_driver2"

# 주기 확인
ros2 topic hz /Odometry
ros2 topic hz /cloud_registered
ros2 topic hz /cam_front/color/image_raw
```
Gate G1 체크:
- 위 3개 hz 명령이 최소 8Hz 이상 안정 출력

### C. TF/동기화 체크
```bash
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo camera_init body
ros2 run tf2_ros tf2_echo map base_link
ros2 run tf2_ros tf2_echo base_link lidar_frame
ros2 run tf2_ros tf2_echo lidar_frame cam_front_color_optical_frame
```
Gate G2 체크:
- `view_frames` 체인에 단절 없음
- `tf2_echo`가 연속 갱신되고 좌표가 비정상 점프하지 않음

### D. 통합 전 최소 헬스체크
```bash
ros2 topic echo /action --once
ros2 topic hz /scout_mini_base_controller/cmd_vel
```
Gate G3/Phase4 체크:
- `/action` 유효 JSON payload 확인
- `cmd_vel`이 제어 주기(목표 10Hz 근처)로 출력

---

## 11) 리스크 및 대응

### R1. TF 축/부호 불일치
- 증상: 이동 방향과 맵/영상 방향 불일치
- 대응: 축 정의표 고정 후 단일 노드에서만 좌표 변환 수행

### R2. 시간동기 불량
- 증상: RGB-포즈 mismatch, frontier 이미지 선택 오류
- 대응: source timestamp 사용, 버퍼 기반 nearest-time 매칭

### R3. Occupancy 품질 저하
- 증상: frontier 난립 또는 없음
- 대응: ray-casting/free-space 업데이트, height filter 튜닝

### R4. 실행 환경 혼용
- 증상: 패키지 충돌/토픽 불일치
- 대응: Humble 기준으로 통일, `safe_realsense_launch.sh`(Jazzy 기준)는 본 파이프라인에서 분리

---

## 12) 최종 Definition of Done

아래를 모두 만족하면 프로젝트 완료로 간주:
- FAST-LIO2 + 3카메라 + Scout 제어가 동일 ROS2 네트워크에서 안정 구동
- slow가 occupancy/frontier 기반으로 subgoal을 선택
- fast가 실 pose + tri-RGB 기반으로 `/action` 생성
- `omninav_control.py`가 cmd_vel로 로봇을 안정 구동
- 실제 환경에서 end-to-end 데모 3회 연속 성공

# Track A - Habitat Dependency Replacement Map (A-2)

작성일: 2026-02-24  
근거 raw list: `docs/realworld/habitat_deps_raw.txt`

## 1) 목적

A-1 실측 결과를 기준으로, 실세계 데모에서 Habitat 의존 항목을 무엇으로 대체해야 하는지 1차 매핑한다.  
이 문서는 "대체 대상 식별"이 목적이며, 구현 상세는 Phase 3에서 확정한다.

## 2) 범위

- 포함: `infer_ovon_slowfast` 실행 경로 + realworld demo에서 직접 쓰는 fast/control 스크립트
- 제외: `infer_r2r_rxr/VLN_CE/**` 학습/시뮬레이션 확장 경로 (초기 데모 비핵심)

## 3) 모델 경로 기준 (현재 합의)

- slow+fast 모델 가중치 위치: `/home/aprl_msi/ayun/OmniNav_setup/OmniNav_Slowfast`
- fast 온라인 스크립트 기본값은 별도(`../OmniNav`)이므로 Phase 3 통합 시 런치/인자 통일 필요

## 4) 대체 매핑표 (1차)

| Habitat 의존 항목 | 코드 근거 | 대체 필요 | Candidate ROS Source(1차) | 비고 |
|---|---|---|---|---|
| `maps.get_topdown_map_from_sim` | `infer_ovon_slowfast/run_nav_ovon_omni.py:96` | Yes | `/omni/occupancy` | top-down map 공급원 교체 |
| `sim.step(action)` | `infer_ovon_slowfast/run_nav_ovon_omni.py:125`, `infer_ovon_slowfast/run_nav_ovon_omni.py:217` | Yes | `/action`->`cmd_vel` 실행 후 `/Odometry` 피드백 | 시뮬레이터 스텝 제거 |
| `agent.get_state()` | `infer_ovon_slowfast/run_nav_ovon_omni.py:107` | Yes | `/Odometry` 또는 `tf` | pose source 통일 필요 |
| `sim.get_sensor_observations()` | `infer_ovon_slowfast/run_nav_ovon_omni.py:137`, `infer_ovon_slowfast/qwen_utils.py:245` | Yes | `/cam_front|left|right/color/image_raw` | bank 입력으로 대체 |
| `maps.to_grid / from_grid / calculate_meters_per_pixel` | `infer_ovon_slowfast/frontier_utils.py:17`, `infer_ovon_slowfast/frontier_utils.py:20`, `infer_ovon_slowfast/frontier_utils.py:33` | Yes | `/omni/occupancy` 기준 좌표변환 유틸 | grid metric 정의 필요 |
| `sim.pathfinder.snap_point / is_navigable / find_path / get_island` | `infer_ovon_slowfast/frontier_utils.py:40`, `infer_ovon_slowfast/qwen_utils.py:225`, `infer_ovon_slowfast/path_utils.py:85`, `infer_ovon_slowfast/run_nav_ovon_omni.py:176` | Yes | occupancy-grid 기반 경로탐색 계층 | 구현 상세는 Phase 3 결정 |
| `habitat_sim.GreedyGeodesicFollower` | `infer_ovon_slowfast/run_nav_ovon_omni.py:178` | Yes | occupancy 기반 waypoint executor | 구현 상세 미확정 |
| `MultiGoalShortestPath / geodesic_distance` | `infer_ovon_slowfast/run_nav_ovon_omni.py:307`, `infer_ovon_slowfast/run_nav_ovon_omni.py:312` | Optional | 경로길이 근사 metric(선택) | 데모 필수 아님 |
| `ShortestPath / geodesic_distance` | `infer_ovon_slowfast/utils.py:24`, `infer_ovon_slowfast/utils.py:28` | Optional | 동일 | 평가/로그용 |
| `sim.step_filter` | `infer_ovon_slowfast/qwen_utils.py:223`, `infer_ovon_slowfast/qwen_utils.py:323` | Yes | occupancy 기반 충돌검사 필터 | 구현 상세 미확정 |
| `agent.set_state(...)` | `infer_ovon_slowfast/qwen_utils.py:231`, `infer_ovon_slowfast/qwen_utils.py:340` | Yes | 실제 로봇은 state set 대신 명령+피드백 | 시뮬레이터 전용 동작 제거 |
| Habitat import in fast agent | `infer_r2r_rxr/agent/waypoint_agent.py:6` | No(즉시) | try/except로 optional 유지 | 온라인 추론 경로 직접 의존 아님 |

## 5) Track A 결론

- A-1에서 예상보다 넓은 Habitat API 세트가 관측됨
- A-2는 "대체 대상 식별"까지 완료
- 경로탐색 구현 상세(A*, snap 정책, fallback 정책)는 Track A에서 확정하지 않고 Phase 3에서 결정

## 6) 즉시 후속 액션

1. `docs/realworld/topic_io_contract_v1.md`를 A-3 1차 계약서로 사용
2. Phase 3 시작 시 본 문서의 `Candidate ROS Source(1차)`를 구현 단위로 분해

## 7) 스크립트 점검 중 발견된 리스크 (해결 완료)

- `infer_ovon_slowfast/utils.py`의 인자 파서 수정 완료:
  - `--model_path`: `int` -> `str`
  - `--name`: `int` -> `str`
  - 누락된 `--type` 인자 추가 (`A-star`, `point-goal`)

검증:
- `python3 -m py_compile infer_ovon_slowfast/utils.py infer_ovon_slowfast/run_nav_ovon_omni.py` 통과

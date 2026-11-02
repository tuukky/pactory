# 검증 결과 (factory_sim v2, PLAN_v2 §10)

## 실행 명령

```powershell
python -m pytest tests -q                 # v1 68 + v2 94 = 162 통과
python -m pytest tests/v2 -q              # v2 94 통과 (약 20초)
C:\Users\CEO\anaconda3\envs\n8n\python.exe -m pytest tests -q   # conda n8n 환경도 162 통과
```

- v2 테스트는 `tests/v2/test_v2_*.py`이며 `conftest.py`를 두지 않는다(v1 `from conftest import` 패턴과 모듈 이름 충돌 방지). 공통 헬퍼는 `tests/v2/v2_helpers.py`.
- 검증 run은 짧은 호라이즌·결정론 설정을 쓰며, 탐색 시드 100~119와 확인 시드 1000~1029·2000~2029를 쓰지 않는다.

## 결과표

| ID | 사례 | 테스트 | 통과 기준 | 결과 |
|---|---|---|---|---|
| V01' | 직선 라인 해석해 | `test_v2_01_serial_line.py` (2) | 병목이 라인 속도를 결정, 군별 시간·time_scale | 통과 |
| V02'·V03'·V18 | 보존·경계·자원 시간 보존 | `test_v2_02_conservation.py` (4) | 상태시간 합 = 관측시간(설비·작업자·AGV 각각), 물량 수지, 저장 디렉토리 재계산 = 메모리 값 | 통과 |
| V04' | 막힘 | `test_v2_04_blocking.py` (2) | 버퍼 1·2에서 blocking-after-service 손계산 | 통과 |
| V05'·V14 | 품질·재검사 재순환 | `test_v2_05_quality_recirc.py` (5) | 불합격 1에서 교착 없음, EOL 2회·RW 1회, reserved 슬롯 의미, K=200·불합격 0.5 스트레스 | 통과 |
| V06' | 병렬 설비 준비 | `test_v2_06_parallel_setup.py` (2) | 설비별 준비 기억, 같은 군 다른 SKU 준비 0 | 통과 |
| V07'·V16 | 재현성·CRN | `test_v2_07_reproducibility.py` (5) | 같은 시드 동일, PYTHONHASHSEED 무관, AGV 2 vs 3에서 설비별 k번째 고장 표본·도착 동일, 수요율 변경 시 같은 표본 | 통과 |
| V08·V09 | 고장·재개, 동시 고장 | `test_v2_08_failures.py` (5) | 완료 시각 손계산, 고장 시계는 가공 중에만, 두 설비가 서로 기다리지 않고 수리, `ttf = 남은 가공` 경계, 확률 표본 평균 | 통과 |
| V10 | AGV | `test_v2_10_agv.py` (4) | 운반 = 공차 이동 + 상차 + 이동 + 하차, 묶음 4, 목적지 예약 한도, 충전 시점 | 통과 |
| **V11** | **중첩 일치** | `test_v2_11_v1_equivalence.py` (5) | (a) 결정론: v1과 주문별 출하시각 오차 1e-9 이하·설비별 Job 선택 순서 동일 (b) v1 확률 처리시간 주입 2시드 일치 (c) 작업자·AGV 계층을 켠 채 퇴화 파라미터(이동 0, 전 숙련, AGV 다수)로 일치 | 통과 |
| V12 | 키트 동기화 | `test_v2_12_kit.py` (2) | 모터 1대 = 같은 주문 부품 3개, 슬롯 ≤ K, 하우징 결품 시 투입 정지 | 통과 |
| V13 | 배치 오븐 | `test_v2_13_oven.py` (4) | 가득/형성 시작 후 900초 규칙(C1), 대기품 일괄 적재, 고장 시 배치 전체 정지, 오븐 2대 형성 비중첩 | 통과 |
| V15 | 작업자 | `test_v2_15_operators.py` (4) | 숙련자 1명이 설비 2대를 번갈아 담당(동시 2설비 없음), 준비만/전체 구분, 고장 중 잔류, 숙련 없는 공정 거부 | 통과 |
| V17 | 재고 | `test_v2_17_inventory.py` (2) | 재고위치 기준 발주 시점, 수지 = 초기 + 입고 − 사용, 재고 ≥ 0 | 통과 |
| V19 | 디스패칭 | `test_v2_19_dispatch.py` (6) | FIFO/EDD/FAMILY 손계산 순서, T_max 초과 강제 선택, 엔진에서 FAMILY가 준비 감소 | 통과 |
| — | 설정 검증 | `test_v2_config.py` (19) | include 해석, override 허용 접두어, 레이아웃 노드 누락·숙련 누락·잘못된 값 거부, assumptions.csv = rev_A | 통과 |
| — | 지표 | `test_v2_metrics.py` (5) | 반열린 관측창, 코호트 리드타임과 목표 판정, 출력버퍼 필수, env_info 재사용, 같은 정차 추가 적재 | 통과 |
| — | 정적 계산 | `test_v2_static_capacity.py` (7) | 손계산 일치, 하우징 공급 상한, 작업자·AGV 부하, fail 0·재작업 없음·회복률 변경 엣지 | 통과 |
| — | 통계·설계 절차 | `test_v2_stats_design.py` (11) | MSER-5, 워밍업 규칙, 보정은 현상 키만, 단일 변경 규칙, 선택 규칙(순위 → 개선 폭), 짝 판정 방향, 목표 판정, 직렬 = 병렬 결과, freeze | 통과 |

## 실험 run 무결성

- 모든 run은 metrics의 `checks`를 남긴다: time_conservation(최대 오차 약 1e-10s), wip_area(온라인 WIP 면적 = 주문별 창 체류 합), mass_balance, inventory_balance, kit_bound, agv_never_blocked.
- `results/v2`의 실험 run 816회 전부 `complete_at_horizon`, 실패 0, 검사 전부 통과.
- Rev.A 1 run 실행 시간: summary 약 8초, full 약 13초(base Python, 단일 프로세스).

## 게이트 준수 (PLAN §14)

- V11 (a)(b)(c) 통과 후 보정·설계 실험을 시작했다.
- 보정 동결(`calibration_20261026T230428_cbf19de2`, 23:05 UTC) 후 첫 설계 비교(`A_diag`, 23:10 UTC)를 실행했다.
- 1단계 최종 Rev.B 동결 후, 2단계는 PLAN §17 사전 등록 뒤 실행했다. 2단계 중 엔진 변경(다중 오븐)은 1대일 때 결과가 같음을 확인하고 테스트를 추가했다.

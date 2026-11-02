# 모델 명세 (factory_sim v2)

PLAN_v2.md(v2.1, 정비공 자원 없음)와 §17(2단계 증산 설계)을 구현한 엔진의 계약과 설계 결정을 기록한다. 모든 입력은 합성 가정이다.

## 1. 구성

- 패키지: `src/factory_sim/` (v1 `src/line_sim/`은 동결, 다음 유틸만 import한다)
  - `line_sim.config`: `ConfigError`, `config_hash`, `load_yaml` (factory_sim.config에서 재노출)
  - `line_sim.randomness`: `lognormal_from_z`, `op_code` (factory_sim.randomness에서 재노출)
  - `line_sim.experiments`: `NO_CLEAR_DIFF`, `make_experiment_id`, `mean_ci`, `paired_diff`, `parse_seeds` (experiments.py), `mean_ci`, `parse_seeds` (calibration.py)
  - `line_sim.run`: `environment_info`, `collect_metadata` (run.py)
  - 테스트 V11은 추가로 `line_sim.engine.run_simulation`·`base_duration`, `line_sim.entities.build_input_plan`을 쓴다.
- 설정: `configs/v2/rev_A.yaml` + `layout.yaml`, `skills.yaml`(include). 개정본 `rev_B.yaml`(s 300→800), `rev_C.yaml`(K 30→45).
- 공장: 설비 21대(로터 R1×3·R2·R3·R4, 스테이터 S1·S2·S3×3·S4×2·S5 오븐, 최종 FA×3·EOL×2·RW·PK), 작업자 8명, AGV 3대, 하우징 구매 재고, 키트 슬롯 K.

| 모듈 | 책임 |
|---|---|
| `config.py` | YAML 로드·include 해석·override(허용 접두어 `layout.nodes`, `operators.skills`, `time_scale`)·검증 |
| `layout.py` | 노드 좌표, 맨해튼 × 우회계수 1.2 거리 |
| `randomness.py` | CRN 키 기반 난수 (`SeedSequence(seed, spawn_key=(DOMAIN, *key))`) |
| `entities.py` | 상태 추적(StateTracker), 입력 Buffer(general/reserved), OutputBuffer, KitArea, 시간가중 집계 |
| `routing.py` | 부품 경로, 합류, EOL⇄RW 재순환, 공정별 작업자 구간·디스패칭 규칙 |
| `failures.py` | 가공시간 기준 고장 시계, 즉시 수리, 남은 시간 보존 |
| `batch.py` | 오븐 배치 형성·가공·하역 |
| `inventory.py` | 하우징 (s, Q) 재고와 원장 |
| `operators.py` | 작업자 풀, 숙련 매트릭스, 최근접 배정, 도보 |
| `agv.py` | 운반 요청, 최근접 배차, 묶음 적재, 목적지 예약, 충전 |
| `dispatch.py` | FIFO / EDD / FAMILY(T_max) 순수 함수 |
| `release.py` | 주문 도착(포아송·explicit), 키트 슬롯 예약 투입 |
| `logging.py` | 이벤트 로그(EVENT_COLUMNS, EventType), 비활성 시 기록 비용 0 |
| `engine.py` | 조립·수동 `env.step()` 루프·교착 감시·불변식 검사(옵션) |
| `metrics.py` | 관측창 지표, 손실 분해, 무결성 검사 |
| `run.py` | 단일 run CLI (summary/full) |
| `static_capacity.py` | 정적 부하 계산 (가설) |
| `warmup.py`, `calibration.py`, `experiments.py` | MSER-5, 보정, 병렬 실험·짝 비교·후보 선택·개정 동결 |

## 2. 엔진 계약

- 시간 단위 초, 가동시간 기준 연속 시간(교대 경계 없음). 1 run = 워밍업 288,000s(80h) + 관측 576,000s(160h), 호라이즌 864,000s.
- 수동 `env.step()` 루프로 호라이즌과 같은 시각의 이벤트까지 처리한다. 상태 `complete_at_horizon`. 이벤트 큐가 비거나, WIP가 있는데 진행 이벤트가 `deadlock_window_s`(기본 86,400s) 동안 없으면 `DeadlockError`로 실패 처리한다.
- 무결성 검사는 두 층이다.
  - **실행 중 불변식**(물량 수지, 부품 위치 유일성, 클래스별 버퍼 용량, 슬롯 ≤ K, 재고 ≥ 0): `run_simulation(..., check_invariants=True)`일 때만 매 시각 끝에서 검사한다. 테스트에서만 켠다.
  - **사후 metrics checks**(time_conservation, wip_area, mass_balance, inventory_balance, kit_bound, agv_never_blocked): 모든 run(실험 포함)이 metrics.json에 남긴다. 실험 run은 실행 중 불변식 검사 없이 이 사후 검사만 거쳤다.
- 0초 지연은 yield하지 않는다(동시각 순서 보존, V11 조건). 고장 처리 내부의 `timeout(0)`만 v1 동치를 위해 예외적으로 유지한다.
- 관측창은 반열린 구간 [w0, w1)이다. 출하·도착·납기 모두 같은 규칙이다.
- 실험은 summary 모드(이벤트 로그 off). full 모드는 events·구간·운반·배치·재고 원장을 저장한다(run당 약 55~90MB).

### 2.1 공정 흐름과 자원 확보

1. 주문 도착 → 백로그. 투입은 원자적: 빈 키트 슬롯 + 하우징 재고가 있으면 슬롯 예약 → 재고 차감 → ROTOR·STATOR Job을 RM, HOUSING을 HW 출력에 생성.
2. 서브라인 내부는 컨베이어(이동 0, 유한 버퍼, blocking-after-service). 서브라인 끝 → KIT, KIT 이후는 AGV.
3. 설비 확보 순서: **Job 확보(=설비 점유) → 작업자 요청 → 준비 → (준비만 공정이면 작업자 해제) → 가공(고장 포함) → (가공 전체 공정이면 PROCESS_END에서 해제) → 하류 put**. 작업자는 Job이 있는 설비에만 배정되고 막힌 설비를 잡지 않는다.
4. FA는 3개 부품이 모두 도착한 슬롯을 디스패칭 규칙으로 가져가며, 그 시점에 슬롯이 해제된다.
5. EOL 합격 → PK → DOCK(출하). 최초 불합격 → RW → 같은 EOL 재검사(reserved 슬롯 우선). 재검사 불합격 → 폐기.

### 2.2 설비 상태와 손실 분해 (7범주)

| 상태 | 손실 범주 | 비고 |
|---|---|---|
| PROCESSING | 가공 | |
| SETUP | 준비 | 품종군 변경 시에만, 설비 첫 Job 0 |
| STARVED | 굶주림 | 오븐 형성 중은 detail `batch_forming` |
| BLOCKED_DOWNSTREAM | 막힘(하류) | 컨베이어 입력 가득, 또는 출력버퍼 가득 + 목적지 여유 없음 |
| BLOCKED_TRANSPORT | 막힘(운반) | 출력버퍼 가득인데 목적지 여유가 있음(AGV 미도착) |
| WAIT_OPERATOR | 작업자 대기 | 배정 대기 + 도보 |
| DOWN | 고장 | 즉시 수리, 세분 없음 |

- 막힘은 목적지 공간·AGV claim·출력버퍼 변화 때마다 재분류한다.
- 작업자 5상태(IDLE/WALK/SETUP/PROCESS/WAIT_REPAIR), AGV 8상태(IDLE/TRAVEL_EMPTY/LOAD/TRAVEL_LOADED/UNLOAD/BLOCKED/TO_CHARGE/CHARGING).
- 설비·작업자·AGV 각각 상태시간 합 = 관측시간(허용오차 1e-6s)을 매 run 검사한다(V18).
- 결품은 설비 상태가 아니라 공장 지표(투입 정지 시간)로 집계한다.

### 2.3 CRN 키

- 처리시간 z: `(order_id, part, op, dur|durN)`, 품질: `(order_id, MOTOR, EOL, fail_u)`·`(…, RW, recover_u)`.
- 고장: `(machine_id, k, ttf|repair)`. ttf = 단위지수 × MTBF × mtbf_scale이라 노후 조건도 같은 표본을 쓴다. 설비를 추가하면 새 ID가 새 스트림을 받는다.
- 도착: `("arrivals", gap|group|sku)`. 간격 = E_i / λ라 수요 +15%는 같은 E_i를 축소한다.
- 배정·배차·디스패칭은 난수를 쓰지 않는다.

## 3. 목표 지표 정의 (D5~D7)

| 지표 | 정의 | 목표 |
|---|---|---|
| 수요 충족률 | 관측창 양품 출하 / 관측창 도착 | ≥ 0.98 |
| 납기 충족률 | 납기가 관측창에 속한 주문 중 납기 이전 양품 출하 비율(폐기 = 미충족) | ≥ 0.95 |
| 리드타임 | 관측창 도착 코호트의 도착→종료 평균. 미완료는 sim_end − 도착(하한) | ≤ 57,600s(16h) |

- 리드타임은 디버깅 W2 결과로 출하분만 평균하던 방식(생존자 편향)에서 코호트 하한으로 바꿨다. 출하 기준 평균·p90도 함께 보고한다.
- 폐기율(약 0.7%)이 공통 난수로 고정되므로, 공급이 충분한 설계에서는 납기 충족률 상한이 약 0.993이다. 개정 간 납기 차이가 0으로 나오는 이유다.

## 4. 판단 항목 결정 (D1~D30, 팀장 채택)

| # | 결정 |
|---|---|
| D1 | AGV는 배차 시 목적지 공간을 예약한다(여유 0이면 배차 안 함). 하차 막힘(AGV BLOCKED)은 0초가 되고 이를 검사한다. **PLAN §3.5 "하차 지점이 꽉 차면 짐을 실은 채 대기"와의 편차**다(교착 방지). |
| D2 | RW 입력 버퍼 무제한(불량 대기 랙). EOL 입력은 general 6 + reserved 2. |
| D3 | EOL→PK는 AGV 운반(그림 기준). |
| D4 | 오븐 챔버(24)는 형성 중 직접 적재, 입력 버퍼 6은 가동 중 대기열. 하역은 출력 8로 순차. |
| D5~D7 | §3 표. |
| D8 | 보정 지표는 부하 기준: DOWN/(PROCESSING+DOWN), SETUP/(SETUP+PROCESSING). |
| D9 | 보정 순서 품질 → 고장 → 준비, 범위 밖 현상만 클래스 공통 배율 1개, 격자 ≤ 5회. |
| D10 | SKU 계수 0.90/0.9667/1.0333/1.10(평균 1), 가공·검사·재작업·포장에만 적용. |
| D11 | FAMILY T_max = 3,600s 고정. |
| D12 | FA FIFO = 키트 완료 시각, EDD = 납기(동률 완료시각). |
| D13 | FA는 준비가 없어 FAMILY는 순서만 바꾼다. |
| D14 | PK 입력 6, RW 입력 무제한. |
| D15 | AGV 충전: 이동+상하차 누적 14,400s 후 현재 운반을 마치고 CHG로 이동, 1,800s 충전, 충전기 무제한. |
| D16 | 상·하차 20s는 정차 1회당. |
| D17 | 유휴 시 마지막 위치에 머문다. 초기 위치 AGV=CHG, 작업자=skills.yaml home. |
| D18 | 작업자 해제: full은 PROCESS_END, setup_only는 준비 종료. |
| D19 | `ttf_left ≤ 남은 가공`이면 그 시점에 고장. 준비·형성·막힘 중 고장 없음. |
| D20 | 준비는 품종군 변경 시에만, 같은 군 SKU 사이 0, 설비 첫 Job 0. |
| D21 | (s, Q)는 재고위치(보유 + 미착) 기준. |
| D22 | 투입 순서 FIFO(납기 = 도착 + 상수라 EDD와 같음). |
| D23 | 호라이즌 240h 절단, 도착은 전 구간 생성. |
| D24 | 워밍업: 보정 시드 MSER-5 d* 최댓값 ≤ 80h면 80h 유지(결과 d* = 0h → 80h). |
| D25 | 버퍼 재배분 대상: 서브라인 입력·출력 버퍼. 제외: EOL reserved, RW 입력, K, 오븐 챔버. |
| D26 | "작업 분할"은 정의된 분할이 없어 후보에서 제외. |
| D27 | 품종 C 증가 조건은 주문 집합이 달라 짝 비교하지 않고 목표 만족만 판정. |
| D28 | 탐색 판정은 20시드 평균, 확인은 30시드 평균과 CI 경계를 따로 표기. |
| D29 | 고장 대상은 설비 21대 전부(3개 클래스), AGV 고장 없음. |
| D30 | 오븐 형성 중 = STARVED(batch_forming), 굶주림의 하위 항목. |
| D31 | 후보 선택 규칙의 추가 조건: primary CI가 개선 방향으로 0을 포함하지 않아도, **다른 목표 지표 중 CI 전체가 악화 쪽인 것이 있으면 부적격**. 실행 전 고정(experiments.select_candidate). 1단계 H3(K 30→45)가 수요 충족 악화 [−0.021, −0.013]로 부적격이 된 원인. |

## 5. 디버깅 단계에서 추가한 결정

### C1. 오븐 최대 대기 900초의 기준 시각

- PLAN은 "최대 대기 900초면 부분 배치"라고만 적었다. 처음에는 첫 품목의 S5 입력 도착 시각 기준으로 구현했다.
- 이 방식에서는 직전 배치 가공 중에 도착한 품목의 기한이 하역 직후 이미 지나 있어, 배치가 입력 6 + 막힌 putter 2 = 약 8개로 굳었다(Rev.A 평균 7.9). 설계 결과가 아니라 모델링 산물이다.
- **채택 규칙**: 900초는 **형성 시작**(빈 챔버에 첫 품목을 적재한 시각, 대기품 일괄 적재 포함)부터 잰다. 챔버가 24개로 차거나 900초가 지나면 가공을 시작한다. 수정 후 Rev.A 평균 배치 14.8.
- 정적 계산에는 이 의미로 실효 배치(이월 8 + 형성 중 유입 4.5 = 12.5, 주기 2,700s → 16.7대/h, 고장 반영 15.4대/h)를 `cap_effective_per_h`로 추가했다.

### 다중 오븐 (2단계 OVEN2 후보용)

- 오븐 +1대 후보를 시험하려고 "배치 오븐 1대" 제한을 없앴다. 스테이션별 형성 토큰을 두어 같은 스테이션에서는 한 번에 한 대만 배치를 형성한다(입력 버퍼를 나눠 먹어 둘 다 작은 배치로 시작하는 것을 막기 위함).
- 오븐 1대일 때 토큰은 항상 즉시 확보되므로 동작이 같다. Rev.B 시드 100 지표가 변경 전과 완전히 같음을 확인했고, 기존 실험은 재실행하지 않았다. 테스트 `test_two_ovens_form_one_at_a_time`.

### 기타

- AGV가 상차하면서 비운 출력버퍼 자리에 막힌 품목이 들어오면, 용량·목적지 여유 안에서 같은 정차에 함께 싣는다(묶음 효율).
- AGV가 출발하는 스테이션은 `output_buffer.capacity`를 반드시 명시해야 한다(조용한 기본값 제거).

## 6. 교착 부재 증명 스케치

자원 순환 대기가 생길 수 있는 곳은 (1) 서브라인 ↔ 키트, (2) AGV 하차, (3) EOL ⇄ RW, (4) 작업자 ↔ 설비다.

1. **키트**: 투입 시 슬롯을 주문에 예약하고 부품은 자기 슬롯으로만 배달된다. KIT 목적지는 예약된 자리라 AGV 예약이 항상 성공한다. 서브라인 WIP는 K 이하라 "로터만 가득 차고 짝 스테이터가 못 들어오는" 상태가 없다.
2. **AGV**: 배차 시 목적지 자리를 예약하므로 실린 AGV가 하차 지점에서 영구히 막히지 않는다(D1). 막힘은 출력버퍼에만 남고, 목적지 자리가 생기면 notify로 다시 배차된다.
3. **EOL ⇄ RW**: EOL 출력 중 PK행은 PK→DOCK(무제한 도크)로, RW행은 RW 입력(무제한, D2)으로 항상 빠진다. 따라서 EOL은 영구히 막히지 않는다. EOL이 진행하면 reserved 슬롯이 비고, RW 출력이 배출되고, RW가 진행한다. 예약 2슬롯만으로는 반례(EOL 2대가 RW행 품목을 들고 막히고 reserved·RW 입출력 모두 가득)가 있어 D2를 추가했다.
4. **작업자**: 작업자는 Job을 이미 확보한 설비에만 배정되고, 막힌 설비는 작업자를 잡지 않으며, 수리에는 작업자가 필요 없다. 작업자 → 설비 방향의 대기 간선만 있어 순환이 없다.

검증: V12(키트), V14(재검사 재순환, 불합격 1·작은 버퍼·K=200 스트레스), `agv_never_blocked` 검사(모든 run), 교착 감시자(모든 run). `results/v2`의 실험 run 816회(보정 36 + 탐색·확인 780)에서 교착·실패 0.

## 7. PLAN과의 편차 요약

| PLAN | 구현 | 이유 |
|---|---|---|
| §3.5 하차 지점 가득 시 AGV가 실은 채 대기 | 목적지 예약으로 하차 막힘 없음 | 교착 방지(D1) |
| §3.2 재검사 예약 슬롯 2개로 교착 방지 | + RW 입력 무제한 | 예약만으로는 반례 존재(D2) |
| §4 오븐 "최대 대기 900초" | 형성 시작 기준 | 배치 8개 고착 산물 제거(C1) |
| §4 목표 "주간 양품 ≥ 주간 수요", "평균 리드타임 ≤ 16h" | 수요 충족률 ≥ 0.98, 코호트 리드타임 하한 | 포아송 변동·폐기·절단 편향(D5, D7, W2) |
| §7 모듈표 | `warmup.py`, `calibration.py` 보조 모듈 추가 | §8·§9.1 구현 |
| §9.2 개정 3~6회 예상 | 1단계 1회 + 2단계(§17) 1회 | 종료 규칙(목표 충족)에 따름 |
| §9.2 채택 기준 "목표 개선 + CI 0 미포함" | + 다른 목표 지표 CI 전체 악화 시 부적격(D31) | 한 지표 개선이 다른 목표를 해치는 후보 배제 |
| §11 "공장 손실 워터폴: 이론 능력 → 실제 출하" | 대시보드 워터폴은 선택 **공정 단위** 근사(공정 이론 능력 − 상태 비율 손실 → 가공 상당 → 실제 양품 출하). 투입 정지·결품을 포함한 공장 단위 워터폴이 아님 | 저장 결과만으로 계산 가능한 근사. 투입 정지 시간은 KPI로 별도 표시 |
| §11 "AGV·작업자 위치 2D 재생" | AGV는 운반 기록으로 선형 보간, **작업자는 도보 보간 없이 마지막 배정 설비에 표시** | 작업자 도보 경로를 이벤트에 저장하지 않음 |

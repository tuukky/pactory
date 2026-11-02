# 모델 명세

모든 수치는 합성 가정이다(`configs/baseline.yaml`, `data/assumptions.csv`). 시간 단위는 초.

## 1. 공정과 경로

```text
외부 주문 풀 (P1 240 / P2 144 / P3 96, 시각 0 준비)
   │ 단일 투입 제어기: 최소 간격 48초 · 전체 WIP ≤ 40 · Q_PREP 공간
   ▼
Q_PREP(8) → PREP(1대) → Q_ASM(8) → ASM(2대, 품종 전환 준비) → Q_TEST(8) → TEST(1대)
   TEST 합격 → Q_PACK(8) → PACK(1대) → 양품 출하(SHIP)
   TEST 불합격 → Q_REWORK(4) → REWORK(1대, 재작업+독립 재검사)
        회복 → Q_PACK,  실패 → 폐기(SCRAP)
```

- 제품 1개 = Job 1개. 재작업 시 새 Job을 만들지 않는다. 재작업은 최대 1회.
- 이동시간 0, 버퍼–공정 사이는 논리적 흐름.
- 처리시간: 검증 모드는 표의 고정값, 확률 모드는 평균=표 값, CV=0.10 로그정규. `time_scale`(예: S3의 ASM ×0.8)은 가공시간에만 곱하고 준비시간에는 적용하지 않는다.

## 2. 버퍼와 막힘

- 버퍼 용량 = **대기 중 제품 수**만. 설비 안 제품은 버퍼 수에 포함하지 않는다.
- 커스텀 `Buffer` (simpy.Store 미사용). FIFO.
- **가공 후 막힘(blocking-after-service)**: 가공이 끝났는데 다음 버퍼가 가득 차면 설비가 제품을 계속 보유하고 BLOCKED 상태로 대기한다. 이 제품은 다음 버퍼의 점유에 포함되지 않는다.
- 버퍼에서 한 개가 빠지면 같은 호출 안에서 가장 오래 막힌 상류 설비의 제품을 바로 채운다(빈 슬롯 가로채기 방지).
- 버퍼가 비어 있고 대기 중인 설비가 있으면 제품을 바로 그 설비에 전달한다(BUFFER_ENTER 직후 같은 시각 BUFFER_EXIT 기록).

## 3. 동시 경합 규칙

- 빈 버퍼를 기다리는 설비가 여러 대면 **대기 시작 시각이 이른 설비 → 설비 ID 생성 순서(ASM_1 → ASM_2)** 순으로 제품을 받는다.
- 막힌 상류 설비가 여러 대면 막힘 시작 순(FIFO)으로 슬롯을 받는다.
- 같은 시각의 도착·종료가 동시에 발생하면 SimPy의 결정론적 이벤트 처리 순서(스케줄 순)를 따른다. 설비 프로세스는 ID 순으로 생성되고 투입 제어기는 그 뒤에 생성된다. 동일 입력·시드·버전에서 순서는 재현된다(`event_seq`).

## 4. 설비 상태

| 상태 | 정의 |
|---|---|
| STARVED | 받을 제품이 없어 대기 (주문 종료 후 무작업 포함) |
| SETUP | 품종 전환 준비 (ASM, 직전 품종 ≠ 현재 품종) |
| PROCESSING | 가공·검사·포장 |
| BLOCKED | 가공 완료 후 다음 버퍼가 가득 차 전달 대기 |
| DOWN | 선택 고장 기능 예약값. **미구현** — 항상 0 |

- 각 ASM 설비는 직전 품종을 독립적으로 기억한다. 최초 작업 준비 0초.
- 상태 구간은 0부터 `sim_end`까지 빈틈없이 기록한다. `sim_end = max(전체 종결시각 또는 제한시각, 관측창 끝)`. 주문이 일찍 끝나도 28,800초까지 STARVED로 연장한다.

## 5. 투입 제어

직전 **실제** 투입 + 48초가 지났고, WIP < 40이고, Q_PREP에 공간이 있을 때만 한 개를 원자적으로 입장시킨다(yield 없이 WIP 증가 → 투입 로그 → Q_PREP 진입). 밀린 투입을 몰아서 하지 않는다. 대기 중인 주문은 외부 주문 풀에 남는다.

## 6. 실행 종료와 관측 경계

- `env.run(until=...)`을 쓰지 않고 수동 `env.step()` 루프를 쓴다. 다음 이벤트 시각이 제한시각(57,600초)보다 **클 때만** 멈추므로 제한시각과 같은 시각의 이벤트는 처리된다.
- 교대 실적은 `terminal_at ≤ 28,800`인 출하(허용오차 없이 `<=`).
- 16시간 안에 모든 주문이 종결되지 않으면 `incomplete_at_limit`. 전체 종결 KPI는 null.

## 7. 난수 키

- 키: `SeedSequence(entropy=seed, spawn_key=(0x4C53, job_id, op_code, draw_code))` → PCG64.
- `op_code`: ORDER 0, PREP 1, ASM 2, TEST 3, REWORK 4, PACK 5 (그 외 이름은 SHA-256 앞 8바이트). `draw_code`: permutation 0, duration_z 1, fail_u 2, recover_u 3. Python 내장 `hash()` 미사용.
- Job ID는 품종 정규 순서(P1 1..240, P2, P3)로 먼저 고정한다. 투입 순서 정책은 Job ID를 다시 부여하지 않는다(묶음 투입 = 무작위 순열을 품종 순위로 안정 정렬).
- Job별로 스테이션마다 표준정규 z 1개, fail_u, recover_u를 미리 뽑는다. 처리시간 = `m·exp(s·z − s²/2)`, `s = sqrt(ln(1+cv²))`. 판정: `fail_u < fail_rate` → 불합격, `recover_u < recover_prob` → 회복.
- 따라서 같은 시드에서 B0·S1–S5는 같은 Job의 기본 시간·품질 난수를 공유한다. 품종 구성이 바뀌는 high_load_mix는 주문 집합이 달라 기본 조건과 짝 비교하지 않는다.

## 8. 로그

`events.csv`: `run_id, event_seq, time_s, event_type, job_id, product_type, machine_id, station_id, buffer_id, buffer_count, from_state, to_state`. `buffer_count`는 이벤트 직후 점유. 이벤트 종류: ORDER_CREATED, JOB_RELEASED, BUFFER_ENTER, BUFFER_EXIT, SETUP_START/END, PROCESS_START/END, INSPECT_PASS/FAIL, REWORK_RECOVERED/FAILED, JOB_SHIPPED, JOB_SCRAPPED, STATE_CHANGE, SIM_END.

## 9. 지표

정의는 PLAN §9를 따른다. 주요 구현 결정:

- 시간평균 WIP·버퍼 점유는 이벤트 기반 계단 함수의 정확한 면적 / 관측시간.
- 납기 관련 지표(교대 내 양품, 납기 충족률, 미출하 분해)는 관측창 끝 w1까지의 **누적** 종결 기준이다(`terminal_at ≤ w1`). 관측창 시작 w0 > 0일 때 창 안 출하 수(`good_shipped_in_window`)는 양품 출하율(`good_rate_per_h`) 계산에만 쓴다.
- 설비 상태 구간은 상태가 바뀌거나 **보유 Job이 바뀔 때** 새로 시작한다(같은 설비의 연속 PROCESSING도 Job별 구간). 보유 Job만 바뀐 경우도 `from_state = to_state`인 STATE_CHANGE로 기록한다.
- 교대 화면의 체류시간은 `completed_only`(관측창 안에서 종결된 Job, 양품+폐기) 기준이며 미완료 수·평균/최대 나이(`28,800 − admitted_at`)를 함께 보고한다.
- 최초 통과율·재작업률(교대): 관측창 안의 INSPECT 이벤트와 Q_REWORK 진입 이벤트 기준.
- 품종별 p90 체류시간은 한 실행 내부 분포이며, 반복 실행 간 CI와 혼용하지 않는다.

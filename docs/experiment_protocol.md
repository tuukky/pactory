# 실험 프로토콜

설정: `configs/experiments.yaml`. 실행 단위는 **독립 시드별 실행 1회**이며, 실행 안 제품 수를 반복 수로 쓰지 않는다.

## 1. 시나리오

| ID | 변경 | 물리적 투입 차이 |
|---|---|---|
| B0 | 기준안 | 없음 |
| S1 | TEST 1→2대 | TEST 설비 +1대 |
| S2 | Q_TEST 8→16 (WIP 40 유지) | Q_TEST 대기 슬롯 +8 |
| S3 | ASM 가공시간 ×0.8 (준비시간 동일) | ASM 가공시간 20% 단축 |
| S4 | PREP 가공시간 ×0.8 | PREP 가공시간 20% 단축 |
| S5 | P1 전량→P2 전량→P3 전량 투입 (주문 수·Job ID·간격 동일) | 없음(운영 정책) |

한 변경안에서 설비 수·속도·불량률을 동시에 바꾸지 않는다.

## 2. 단계

1. **탐색(discovery)**: 시드 100–119(20회), B0·S1–S5, 기본 조건.
   `python -m line_sim.experiments --matrix configs/experiments.yaml --phase discovery`
2. **조합안**: 탐색 결과를 보고 최대 2개만 `combinations`에 정의한다(예: `C1: {components: [S1, S3], rationale: "..."}`). 선택 근거(rationale)를 반드시 적고, 구성 요소의 단일 변경 결과도 함께 보고한다. 구성 요소 간 override 키 충돌·3개 이상 조합은 오류.
   조합 탐색: `--phase discovery --scenarios B0,C1`
3. **후보 동결**: 최종 후보 1개를 `confirmation_candidate`에 기록(또는 `--candidate`). 이후 후보를 다시 튜닝하지 않는다.
4. **독립 확인(confirmation)**: 시드 1000–1029(30회), B0 vs 후보, 조건 3개.
   `python -m line_sim.experiments --matrix configs/experiments.yaml --phase confirmation --candidate <ID>`
   확인 결과를 보고 후보를 다시 튜닝했다면 새 확인 시드를 분리한다(더 이상 독립 확인이 아님).

## 3. 민감도 조건

| 조건 | 내용 | 비교 |
|---|---|---|
| base | 50/30/20, CV 0.10 | B0 vs 후보 |
| high_load_mix | 30/30/40 (P1 144, P2 144, P3 192), 총 480 | 같은 조건의 B0 vs 후보 (기본 조건과 짝 비교 안 함) |
| high_cv | 기본 비율, CV 0.25 | 같은 조건의 B0 vs 후보 |

## 4. 통계

- 각 시나리오·조건: 평균과 95% t 신뢰구간 (`scipy.stats.t.ppf(0.975, n−1)`), n < 2이면 CI 없음.
- 짝 비교: 같은 시드끼리 `d_i = 후보_i − B0_i`, 평균과 95% t CI. 한쪽이 실패·null인 시드는 `dropped_seeds`에 기록.
- 개선율: 짝을 이룬 시드의 B0 평균이 0보다 클 때만 `(후보 평균 − B0 평균) / B0 평균 × 100`.
- **해석 방향 주의**: `mean_d`와 `improvement_pct`는 부호만 계산하며 "좋아짐"을 판정하지 않는다.
  - 높을수록 좋은 지표: `shift.good_shipped`, `shift.overall_otd_rate`, 품종별 `otd_rate`.
  - **낮을수록 좋은 지표**(음수 d·음수 개선율이 개선): `shift.avg_wip`, `shift.unshipped.*`(total/unreleased/wip/scrapped), `full_run.completion_time_s`, `full_run.line_time_mean_s`, `full_run.order_age_mean_s`, `full_run.scrapped_total`.
- CI가 0을 포함하면 "명확한 차이를 확인하지 못함"으로 표기한다.
- 탐색 후보 중 최대값을 고른 선택 편향은 독립 확인 시드로 완화한다.
- 20회·30회는 일정상 기본 예산이며 충분한 표본 수를 보장하지 않는다.

통계 대상 지표: `shift.good_shipped`, `shift.overall_otd_rate`, 품종별 `otd_rate`, `shift.avg_wip`, 미출하 분해(total/unreleased/wip/scrapped), `full_run.completion_time_s`, `full_run.line_time_mean_s`, `full_run.order_age_mean_s`, `full_run.scrapped_total`.

## 5. 산출물과 기록

- `results/<experiment_id>/experiment.json`: 매트릭스 해시, 기준 설정 경로, 시드, 시나리오 설명·물리적 차이, git 커밋·dirty 여부, 패키지 버전.
- `runs_index.csv`: run별 상태(complete / incomplete_at_limit / failed), 오류 메시지, 실행시간.
- `summary.csv`, `summary_by_product.csv`, `paired.csv`.
- 실패한 run은 통계에서 조용히 빼지 않고 `n_failed`로 표시한다. 기존 experiment-id 디렉토리는 덮어쓰지 않는다.
- UI에서 임의로 바꾼 단일 실행(`results/ui_adhoc/`)은 보고서 프로토콜 결과가 아니다.

## 6. 실행 기록

| 단계 | experiment ID | 일시 | 커밋 | 비고 |
|---|---|---|---|---|
| discovery | TBD | TBD | TBD | |
| 조합 탐색 | TBD | TBD | TBD | |
| confirmation | TBD | TBD | TBD | |

# v4 dev 조정 기록 (PLAN_v4 §4.3, §6)

작성 시작: 2026-10-30 / 대상: LM-O dev 50장(366대상)만. eval 분할은 어떤 조정에도 쓰지 않았다(로드·평가·예측 없음).
측정 PC: AMD Ryzen 7 3800X (8코어 16스레드), Windows 11, conda env `n8n`, OMP_NUM_THREADS=16, cv2.setNumThreads(1).

원칙: 아래 "사전 등록" 절은 **해당 실행 결과를 보기 전에** 적었다. 결과를 본 뒤 규칙을 바꾸면 "사후 변경"으로 표시하고 사유를 남긴다.

---

## 1. 사전 등록 (결과 보기 전, 2026-10-30)

### 1.1 색 설정과 최소 k — `color-rank-diag`

- 실행: `run_v4 color-rank-diag --config configs/v4/lmo_v41.yaml` (분할 + 색 순위만, 정합 없음).
  분할은 v4.1 경로와 같다(프레임마다 `o3d.utility.random.seed(seed + im_id)` 후 segment_scene, extent 필터).
- 대상 클러스터: extent 필터 통과 클러스터 중 점의 50% 이상이 정답 가시 마스크 안인 것 (v3 `seg_hit`과 같은 정의).
- recall@k(설정 s) = 대상 클러스터가 색 순위 상위 k 안에 있는 대상 수 / **전체 대상 수**(366, 대상 클러스터가 없는 대상 포함).
- 격자 (3·2·3·2·2·2 = **144 설정**): h_bins {12, 18, 36} × s_bins {1, 2} × s_min {30, 45, 60} × achrom_v_bins {1, 3}
  × h_smooth_sigma {0, 1} × similarity {bhattacharyya, intersection}. min_px = 30, 모델 표본 20,000점(시드 = obj_id) 고정.
  k ∈ {3, 5, 8, 20}.
- **선택 규칙**: ratio@k = recall@k / recall@20 (같은 설정).
  1. k ∈ {3, 5, 8} 순서로, ratio@k ≥ 0.95 인 설정이 하나라도 있는 **최소 k**를 k_min으로 한다.
  2. 그 k_min에서 ratio ≥ 0.95 인 설정 중 recall@k_min 최고를 고른다. 동률은 격자 나열 순서(위 순서의 사전식, 앞쪽 우선).
  3. k ∈ {3, 5, 8} 어디에서도 0.95를 넘지 못하면 k_min = 20 (후보 축소 없음), 설정은 recall@8 최고.
- 고른 색 설정을 `lmo_v41.yaml color.hist / color.similarity`에 쓰고 덤프 실행 전에 고정한다.

### 1.2 게이트 값과 top_k — `eval-dev --dump-candidates --compute-all --top-k 20` → `replay-tune`

- 덤프: 1.1의 색 설정으로 색 순위 상위 20개 후보를 모두 정합하고, 게이트와 무관하게 모든 검사 원값을 기록한다
  (tol_free 4값의 위반 비율, h_tol × achrom_s_max_obs 6조합의 색 일치 비율). 시드 = seed + im_id·100 + obj_id + 원래 클러스터 idx
  → 상위 k < 20 실행은 덤프의 앞 k개와 같은 계산이다(재생 = 실시간, 테스트 06).
- 고정값: tol_match_m 0.010, depth.min_valid_px 50, color_match.s_min_model 40, s_min_obs 30, color_match.min_px 50.
- 격자 (**7,680 조합**, k 값이 4개일 때): top_k {3, 5, 8, 20} 중 **k ≥ k_min** × tol_free_m {0.010, 0.015, 0.020, 0.030}
  × max_violation {0.05, 0.1, 0.15, 0.2, 0.3} × h_tol {10, 15, 20} × min_color_match {0.3, 0.4, 0.5, 0.6}
  × achrom_s_max_obs {60, 90} × min_vis_relaxed {0.2, 0.3} × min_match_px {50, 100}.
- 지표 (재생): 대상별 채택 = select_candidate 규칙(최종 OK 중 fitness × 가시 비율 최고, 없으면 전체 최고).
  ADD(-S) 성공률, 오채택률(채택 & 오답 / 전체).
- 정답 거부 증가(lost_correct_rate) = 같은 k에서 v3 게이트만 쓴 재생에서는 정답(채택 & ADD(-S) 통과)이었는데 이 조합에서는
  정답이 아닌 대상 수 / 전체 대상.
- **선택 규칙**: lost_correct_rate ≤ 0.02 인 조합 중 (ADD(-S) 성공률 − 오채택률) 최대.
  동률: 작은 k → 높은 ADD(-S) → 낮은 오채택률 → 격자 나열 순서.
- 과적합 주의: 366대상에서 7,680조합을 고르므로 dev 수치는 낙관적이다. 거친 격자만 쓰고, eval-final(Part B)로만 최종 판단한다.

### 1.3 확인 실행

- 고른 값을 `lmo_v41.yaml`에 쓰고 `eval-dev --config configs/v4/lmo_v41.yaml`을 **단독으로** 실행해 dev 결과·처리시간을 잰다
  (재생 결과와 같은 값이어야 한다. 다르면 원인을 기록).
- 참고 기준: `eval-dev --config configs/v4/lmo_v41.yaml --seg gt_mask` (gt_mask + v4.1 게이트 상한, 가설 아님).

---

## 2. 결과

### 2.1 color-rank-diag (`results/v4/color-rank-diag`, 2026-10-30)

- 대상 366, 격자 144 설정, OMP 16. 대상 클러스터가 extent 통과 클러스터 어딘가에 있는 비율 76.0%.
- z순(기준선 순서) recall: @3 13.9%, @5 21.3%, @8 34.7%, @20 63.9%.
- 색 순위 recall (설정별 최고): @3 60.9%, @5 68.0%, @8 73.0%; recall@20은 설정에 따라 66.9~75.4%
  (extent 통과 클러스터가 20개를 넘는 프레임에서 색 순위 상위 20이 z순 20보다 대상을 더 잘 포함).
- ratio@k ≥ 0.95 인 설정 수: k=3 0개(최고 0.829), k=5 0개(최고 0.918), k=8 23개(최고 0.970) → **k_min = 8**.
- 선택(규칙 1.1 그대로): **grid_idx 12 = h_bins 12, s_bins 1, s_min 45, achrom_v_bins 3, h_smooth_sigma 0, bhattacharyya**
  (recall@3 58.2%, @5 67.5%, @8 73.0%, @20 75.4%, ratio@8 0.967).
- 경향(참고): 무채색 V 3구간이 1구간보다 recall@5 평균 +15.7%p (can·eggbox·glue 같은 무채색 물체 구별), H 12구간 > 18 > 36,
  bhattacharyya > intersection.
- `configs/v4/lmo_v41.yaml`의 color.hist / similarity를 위 값으로, top_k를 8로 바꿨다. replay-tune의 top_k 격자는 {8, 20}
  (k ≥ k_min) → 격자 크기 **3,840 조합**.

### 2.2 후보 덤프 (`results/v4/eval-dev_v41_dump_k20`)

- `eval-dev --config lmo_v41.yaml --dump-candidates --compute-all --top-k 20` (색 설정 2.1, 게이트 값은 초안), OMP 16 단독.
- 366대상, 후보 7,320개. 이 실행 자체의 결과(k=20, 초안 게이트): ADD(-S) 33.1%, 오채택률 51.4%, 영역 확보율 75.7%,
  처리시간 중앙값 9.65 s (덤프용 격자 변형 계산 포함이라 처리시간 비교에는 쓰지 않는다).

### 2.3 replay-tune (`results/v4/replay-tune`)

- 격자 3,840 조합 (top_k {8, 20}), 제약 lost_correct_rate ≤ 0.02를 만족하는 조합 2,472개.
- 참고(같은 k, v3 게이트만 = 색 순위만): k=8 ADD(-S) 31.1%, 오채택률 40.2%, 채택률 71.3% / k=20 30.6%, 54.1%, 84.7%.
- **선택 (규칙 1.2 그대로)**: top_k **8**, tol_free_m **0.015**, max_violation **0.10**, h_tol **15**, min_color_match **0.6**,
  achrom_s_max_obs **90**, min_vis_relaxed **0.3**, min_match_px **50**
  → 재생 ADD(-S) **33.1%**, 오채택률 **27.9%**, 채택률 60.9%, 정밀도 54.3%, lost 1.37%p, gained 3.28%p (같은 k 대비).
  min_match_px 50과 100은 결과가 같아 격자 순서로 50.
- 같은 목적함수 최고값의 k=20 조합: ADD(-S) 33.1%, 오채택률 33.6% → k=8이 오채택률에서 낫다(후보 축소가 오채택도 줄임).
- 주의: min_color_match 0.6은 **격자 상한값**이다. 사전 등록 격자를 넓히지 않았다(결과 본 뒤 격자 확장 = 사후 조정이므로).
  상위 조합 간 차이(ADD(-S) 0.5~1%p)는 2.1·verification §2.3에서 본 16스레드 실행 간 잡음(1~2%p) 안이다.
- 제약 없이 목적함수 최고는 0.104(정답 거부가 2%p 넘게 늘어나는 조합)였다. 제약 때문에 고르지 않았다.
- `configs/v4/lmo_v41.yaml`에 위 값을 반영했다.

### 2.4 확인 실행 (`results/v4/eval-dev_v41`, 단독, OMP 16) — 최종 v4.1 dev 결과

| 지표 | 기준선 R1 | v4.1 재생 예측 | **v4.1 확인 실행** | 짝지은 부트스트랩 (v41 − R1, 95% CI) |
|---|---:|---:|---:|---|
| ADD(-S) 성공률 | 32.2% | 33.1% | **34.4%** | +2.2%p [−3.3, +7.6] — 0 포함 (명확한 차이 확인 못함) |
| 오채택률 | 54.6% | 27.9% | **26.0%** | −28.7%p [−33.7, −23.7] |
| 채택 정밀도 | 37.1% | 54.3% | **57.0%** | +19.9%p [+13.8, +25.9] |
| 채택률 | 86.9% | 60.9% | 60.4% | — |
| 처리시간 중앙값 / p95 | 10.32 s / 15.45 s | — | **4.04 s / 6.77 s** | −6.28 s [−6.75, −5.92] |
| 대상 영역 확보율 | 67.8% (z순 20후보) | — | 72.7% (색 상위 8) | +4.9%p [−1.4, +10.0] |

- 실패 코드: OK 221, LOW_VISIBILITY 68, DEPTH_INCONSISTENT 48, COLOR_MISMATCH 24, LOW_FITNESS 5, OTHER 0.
- 단계별 처리시간 중앙값: RANSAC 3.39 s, 전처리 0.20 s, ICP 0.17 s, DBSCAN 0.14 s, 색 0.019 s, 깊이·색 검사 0.018 s.
- 재생 예측과 확인 실행의 차이(ADD(-S) +1.4%p, 오채택률 −1.9%p)는 16스레드 RANSAC 실행 간 잡음이다
  (재생은 덤프 실행의 정합 결과를 쓰고, 확인 실행은 정합을 다시 한다; verification §2.1).
- 부트스트랩: `results/v4/eval-dev_v41/paired_bootstrap_vs_R1.csv` (프레임 50, 10,000회, seed 20261030). **dev 결과이며 가설 판정이 아니다**
  (H1·H4 판정은 eval-final에서만).

### 2.5 참고 기준: gt_mask + v4.1 게이트 (`results/v4/eval-dev_v41_gt_mask`)

| | gt_mask + 기준선 게이트 (v3 `eval-bop_gt_mask_20261029T233048`) | gt_mask + v4.1 게이트 |
|---|---:|---:|
| ADD(-S) | 58.5% | **60.9%** |
| 오채택률 | 13.4% | **5.7%** |
| 정밀도 | 81.4% | 91.4% |
| 채택률 | 71.9% | 66.7% |
| 처리시간 중앙값 | 0.87 s | 0.93 s |

v4.1(dbscan) 34.4%와 gt_mask + v4.1 상한 60.9%의 격차 26.5%p가 남은 분할 손실이다 (v4.2의 대상). 참고 기준이며 가설이 아니다.

---

## 3. v4.2 융합 (Part B) — 사전 등록 (결과 보기 전, 2026-10-30)

학습 데이터는 합성(`train_synth`)만, 모델 선택은 합성 검증(`val_synth`)만 쓴다. LM-O dev는 아래 최소 조정과 최종 점검에만, eval은 쓰지 않는다.

### 3.1 고정 규칙

- **conf** (dev로 정하지 않음): `seg_model val-synth`의 합성 val 클래스 평균 Mask F1 곡선 최대 conf를 0.05 단위로 반올림
  (`configs/v4/train_seg.yaml` fusion_threshold_rule). NMS iou 0.7, imgsz 640, retina_masks.
- dev 예측은 학습 종료 후 **한 번** 만든다 (`seg_model predict --split dev`, 예측 파일에 conf ≥ 0.05 저장, 융합에서 conf로 거름).
- 신뢰도 검사 값(color, checks)은 `lmo_v41.yaml` 그대로 (재조정하지 않는다).
- **dev 최소 격자 (6 조합)**: erode_px {0, 2, 4} × max_instances_per_class {1, 3}.
  - erode_px마다 `eval-dev --config lmo_v42.yaml --compute-all --dump-candidates` 1회 (max_instances 3, 조기 종료 없이 3개 모두 정합·검사).
  - `replay-v42`로 k ∈ {1, 3} 재생: 대상별로 순위 < k 후보 중 처음 최종 OK를 채택 (`select_first_pass`와 같은 규칙).
- **선택 규칙**: (ADD(-S) 성공률 − 오채택률) 최대. 동률은 작은 erode_px → 작은 max_instances_per_class.
  기준 비교 대상이 없으므로 v4.1의 "정답 거부 증가 ≤ 2%p" 제약은 두지 않는다.
  상위 조합 간 차이가 1~2%p면 16스레드 실행 간 잡음(verification §2.3) 안이라는 점을 함께 적는다.
- **확인 실행**: 고른 값으로 `eval-dev --config lmo_v42.yaml`을 **단독**(OMP 16, 렌더링·학습과 동시 실행 금지)으로 1회 →
  v4.2 dev 결과·처리시간. 기준선 R1, v4.1(`eval-dev_v41`)과 프레임 짝지은 부트스트랩 (dev 결과이며 가설 판정 아님).
- 진단(가설 아님): dev 분할 품질 = 대상별 같은 클래스 마스크(conf 이상) 중 정답 가시 마스크와 최고 IoU, 검출률(IoU ≥ 0.5).
  참고 기준 gt_mask + v4.1 게이트(= v4.2 게이트, 검사 값 동일)는 `eval-dev_v41_gt_mask`(60.9%)를 그대로 쓴다.
- dbscan 대체 경로(`fallback_dbscan`)는 주 결과에 쓰지 않는다. dev 참고 실행은 시간이 남을 때만 한다.

### 3.2 실행 전 고정값 (2026-10-31, 격자 실행 전 기록)

- 가중치: `results/v4/seg-train_n/weights/best.pt` (yolo11n-seg, 합성 학습), sha256 `a1ca23be7349a026181fc59008fb03776be11335499cb3e58a10f9feca91d736`.
- `seg_model val-synth` (`results/v4/seg-train_n/val_synth.json`, 합성 val_synth 500장 3,169 인스턴스):
  mask mAP50 0.988, mask mAP50-95 0.904, box mAP50-95 0.945. 클래스별 mask mAP50-95 0.851(glue) ~ 0.955(driller).
  클래스 평균 Mask F1 최대 conf 0.5576 (F1 0.977) → **conf 0.55** (규칙 3.1 그대로, dev 미사용).
- dev 예측 1회: `seg_model predict --split dev --out results/v4/seg-predict_dev` (50프레임, conf ≥ 0.05, 추론 중앙값 22.1 ms/프레임, GTX 1060).
- `configs/v4/lmo_v42.yaml`에 weights, weights_sha256, conf 0.55, predictions.dev를 기록했다.
- 격자 실행 사본: `results/v4/tune-v42_configs/lmo_v42_e{0,2,4}.yaml` (erode_px만 다르고 나머지는 lmo_v42.yaml과 같음, max_instances_per_class 3).
  세 덤프 실행은 **순차** 실행(OMP 16)했다. 처리시간은 선택에 쓰지 않는다.
- 판단(보수적 해석): max_instances_per_class k=1 재생은 덤프의 순위 0 후보만 보는 것이므로 실시간 k=1 실행과 같은 계산이다
  (후보 시드 = seed + rank, compute_all과 무관). 세 erode 값 × k {1, 3} = 6 조합 모두 `replay-v42` 한 번으로 평가한다.
- 주의: 격자 덤프 실행(`eval-dev_v42_dump_e{0,2,4}`) 동안 실행 환경에 외부 CPU·GPU 부하가 있었을 수 있다.
  따라서 격자 실행의 처리시간은 **유효하지 않다** (선택은 ADD(-S) − 오채택률만 쓰므로 영향 없음). v4.2 처리시간은 4단계 단독 확인 실행만 쓴다.

### 3.3 일시 중지 (2026-10-31)

- 완료: 1단계 `seg_model val-synth` (conf 0.55, lmo_v42.yaml에 weights·sha256·conf·predictions.dev 기록), 2단계 `seg_model predict --split dev` (`results/v4/seg-predict_dev`).
- 중단: 3단계 erode 격자 덤프. e0 실행 중 중단(e2도 중단), e4는 시작 안 함. 격자 결과는 하나도 보지 않았다 (replay-v42 미실행).
- 재개 시: 부분 결과 `results/v4/eval-dev_v42_dump_e*` 폴더와 로그를 **모두 지우고** e0 → e2 → e4를 처음부터 다시 실행한 뒤 `replay-v42 --ks 1 3`.
  이후 4단계(단독 확인 실행 `eval-dev_v42`), 5단계(동결·freeze-manifest·pytest) 순서는 그대로.
- 재개 (2026-10-31): 부분 결과 폴더·로그를 지우고 e0 → e2 → e4를 처음부터 순차 실행했다 (재개 시 외부 부하 없음).

### 3.4 결과 — 융합 격자 (`results/v4/eval-dev_v42_dump_e{0,2,4}` → `results/v4/replay-v42`)

| erode_px | k (max_instances) | ADD(-S) | 오채택률 | 채택률 | 정밀도 | 영역 확보율 | ADD − FA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | **1** | 28.7% | 5.5% | 34.2% | 84.0% | 41.3% | **0.232** |
| 0 | 3 | 31.1% | 9.3% | 40.4% | 77.0% | 47.8% | 0.219 |
| 2 | 1 | 29.2% | 6.6% | 35.8% | 81.7% | 41.5% | 0.227 |
| 2 | 3 | 31.4% | 10.4% | 41.8% | 75.2% | 48.1% | 0.210 |
| 4 | 1 | 27.9% | 6.8% | 34.7% | 80.3% | 41.8% | 0.210 |
| 4 | 3 | 30.1% | 10.4% | 40.4% | 74.3% | 48.4% | 0.197 |

- **선택 (규칙 3.1 그대로)**: erode_px **0**, max_instances_per_class **1** (목적함수 0.232, 동률 없음). `configs/v4/lmo_v42.yaml`에 반영.
- 상위 조합 간 차이(0.232 vs 0.227 vs 0.219)는 16스레드 실행 간 잡음(ADD 1.4%p, 오채택 0.8%p) 안이다. k=3은 ADD를 +2.2~2.5%p 올리지만
  오채택을 +3.7~3.9%p 더 올려 목적함수에서 진다.
- 세 erode 모두 NO_DETECTION 134/366 (36.6%): conf 0.55 이상 같은 클래스 마스크가 없는 대상. 격자로는 바뀌지 않는다 (conf는 dev로 정하지 않음).

### 3.5 확인 실행 (`results/v4/eval-dev_v42`, 단독, OMP 16, PC 유휴) — 최종 v4.2 dev 결과

| 지표 | 기준선 R1 | v4.1 | v4.2 재생 예측 | **v4.2 확인 실행** |
|---|---:|---:|---:|---:|
| ADD(-S) 성공률 | 32.2% | 34.4% | 28.7% | **29.2%** |
| 오채택률 | 54.6% | 26.0% | 5.5% | **5.7%** |
| 채택 정밀도 | 37.1% | 57.0% | 84.0% | **83.6%** |
| 채택률 | 86.9% | 60.4% | 34.2% | 35.0% |
| 처리시간 중앙값 / p95 | 10.32 s / 15.45 s | 4.04 s / 6.77 s | — | **0.69 s / 1.12 s** |
| 대상 영역 확보율 | 67.8% | 72.7% | 41.3% | 41.3% |

짝지은 프레임 부트스트랩 (50프레임, 10,000회, seed 20261030; `results/v4/eval-dev_v42/paired_bootstrap_dev.csv`):

| 비교 | ADD(-S) | 오채택률 | 정밀도 | 처리시간 중앙값 | 영역 확보율 |
|---|---|---|---|---|---|
| v4.1 − 기준선 | +2.2%p [−3.3, +7.5] (0 포함) | −28.7%p [−33.7, −23.7] | +19.9%p [+13.8, +25.9] | −6.28 s [−6.75, −5.92] | +4.9%p [−1.4, +9.9] |
| v4.2 − v4.1 | −5.2%p [−11.0, +0.8] (0 포함) | −20.2%p [−24.4, −16.1] | +26.6%p [+18.7, +34.6] | −3.35 s [−3.61, −2.97] | −31.4%p [−37.8, −24.3] |
| v4.2 − 기준선 | −3.0%p [−8.4, +2.4] (0 포함) | −48.9%p [−53.6, −44.6] | +46.5%p [+39.9, +53.3] | −9.63 s [−10.12, −9.12] | −26.5%p [−32.0, −20.8] |
| v4.2 − 0.5·기준선 (H3 형식) | | −21.6%p [−24.4, −19.0] | | | |

- 실패 코드: OK 128, NO_DETECTION 134, LOW_FITNESS 44, LOW_VISIBILITY 27, DEPTH_INCONSISTENT 23, COLOR_MISMATCH 10.
- 재생 예측과 확인 실행 차이(ADD +0.5%p, 오채택 +0.2%p)는 실행 간 잡음 범위다.
- 처리시간 주의: 중앙값에 NO_DETECTION 대상(정합 없이 ≈ 22 ms, 36.6%)이 섞여 있어 낮아 보인다. 단계별 중앙값: RANSAC 0.65 s, seg 추론 22 ms(프레임당).
- 진단 (가설 아님, `seg_quality.csv`): conf 0.55 이상 같은 클래스 마스크가 있는 대상 63.4%, 최고 IoU ≥ 0.5 검출률 46.4%(순위 1 마스크 기준 39.9%).
  물체별 검출률: duck 97.5%, cat 64.4%, eggbox 52.2%, holepuncher 48.0%, can 38.0%, ape 33.3%, driller 32.0%, glue 10.0%.
  → v4.2의 ADD 손실은 합성→실측 분할 격차(sim-to-real gap, 검출 누락)에서 온다. 채택한 것의 정밀도는 높다(83.6%).
- compare-runs: `results/v4/eval-dev_v42/compare_vs_v41.json`, `compare_vs_baseline_R1.json`.
- dev 결과이며 가설 판정이 아니다 (H2·H3 판정은 eval-final에서만). dev 결과를 보고 conf·격자·규칙을 바꾸지 않았다.

### 3.6 동결 (2026-10-31)

- `lmo_v41.yaml`, `lmo_v42.yaml`에 `frozen: true` (기준선은 이미 동결). `run_v4 freeze-manifest` → `configs/v4/frozen_manifest.json`.
- 전체 pytest (n8n): 272 통과, 1 skip, 1 xfail.

# v4 검증 결과 (PLAN_v4 §7) — Part A (S0~S3: 환경, v4.0 기준선 동결·재현, v4.1)

작성일: 2026-10-30 / 측정 PC: AMD Ryzen 7 3800X (8코어 16스레드), GTX 1060 6GB, Windows 11.
CPU 파이프라인: conda env `n8n` (open3d 0.19.0, opencv-python 4.12.0.88, numpy 2.2.6; v3와 같음, 신규 설치 없음).

> LM-O 수치는 모두 **dev 50장(366대상)** 결과다. eval 분할은 로드·평가·예측하지 않았다 (eval-final은 Part B, 세 버전 동결 후 1회).
> LM-O는 **실측** 데이터다(합성 아님). 합성 CAD·간이 장면 테스트 결과는 합성이다.

## 0. 사전 등록 (결과 보기 전, 2026-10-30)

### 0.1 H3 판정 규칙 (PLAN_v4 §6.3, 구현계획 §9-6)

- H3: "v4.2의 오채택률은 기준선의 절반 이하다."
- 판정: eval-final의 같은 eval 50프레임에서 FA = 오채택률(채택 & 오답 / 전체 대상).
  **점추정 FA_v42 ≤ 0.5 · FA_base 이고, 프레임 단위 짝지은 부트스트랩(10,000회, seed 20261030, 백분위 95% CI)으로 구한
  (FA_v42 − 0.5 · FA_base)의 CI 상한 < 0 이면 "지지".**
  점추정만 만족하고 CI 상한 ≥ 0 이면 "명확한 차이를 확인하지 못함", 점추정이 만족하지 않으면 "지지하지 않음".
- 구현: `stats.paired_frame_bootstrap(dfs, [("baseline", "v42", 0.5)], ["false_accept_rate"])` (계수 쌍 b − c·a).
- dev 결과로 이 규칙을 바꾸지 않는다.

### 0.1b eval-final 실행 조건과 H1·H2·H4 판정 규칙 (Part B, 동결 전 기록 2026-10-30)

팀장 결정 (2.1의 TODO 해소):
- **모든 버전을 eval에서 한 번만** 실행한다: `run_v4 eval-final` 1회, `OMP_NUM_THREADS=16`, 시작 시 `--o3d-init-seed 0`
  (Open3D 전역 난수 시드를 프로세스 시작 후 한 번 고정 — 기준선의 첫 프레임 분할 비결정성 제거; 16스레드 RANSAC 잡음은 남는다),
  **단독 실행**(학습·렌더링·pytest·다른 평가와 동시 실행 금지).
  하나의 프로세스에서 eval 프레임을 1회 순회하고 프레임마다 baseline → v41 → v42 → gt_mask(기준선 게이트) → gt_mask(v4.1 게이트) 순서로 계산한다.
- 보고서에는 eval 차이와 함께 **dev R1/R2 실행 간 잡음 폭**(같은 설정 16스레드 2회: ADD(-S) 1.37%p, 오채택률 0.82%p,
  대상별 일치율 80.3%; 2.3)을 적는다.
- **eval 결과를 본 뒤 재실행하지 않는다.** 잠금 파일(`results/v4/eval-final.lock`, `open(path, "x")`)이 있으면 eval-final은 거부된다.
  중간 실패도 재실행하지 않고 실패로 보고한다.

가설 판정 (eval-final 같은 50프레임, 프레임 짝지은 부트스트랩 10,000회, seed 20261030, 백분위 95% CI; `run_v4.hypothesis_verdicts`):
- 성분 판정: 차이(b − a)의 CI가 0을 포함하면 "명확한 차이를 확인하지 못함", 0을 벗어나고 가설 방향이면 "지지", 반대 방향이면 "지지하지 않음".
- H1 (v41 vs 기준선): ADD(-S) 증가와 오채택률 감소 두 성분 모두 "지지"면 지지, 하나라도 반대 방향 유의면 "지지하지 않음", 그 밖은 "명확한 차이를 확인하지 못함".
- H2 (v42 vs v41): ADD(-S) 성분으로 판정. "주된 이유 = 대상 영역 확보율 증가"는 영역 확보율 차이의 성분 판정을 함께 보고한다(근거).
- H3: 0.1의 규칙 그대로.
- H4 (v41 vs 기준선): 처리시간 중앙값 감소 성분으로 판정.

### 0.2 기준선 재현 판정 (구현계획 §7)

- A (정확 일치): OMP_NUM_THREADS=1, 단독. v3 `run eval-bop --limit 3` vs `run_v4 eval-dev --config configs/v4/lmo_baseline.yaml --limit 3`
  → 대상별 code, fitness, rmse_mm, add_mm **정확히 같음** (시간 열 제외).
- B (전체 dev, 16스레드 단독, R1) vs `results/v3/eval-bop_dbscan_20261029T233530`:
  대상 집합 동일, ADD(-S)·채택률·오채택률 |Δ| ≤ 3.0%p, 대상별 (채택, 정답) 일치율 ≥ 90%, 짝지은 부트스트랩 CI ∋ 0.
  실패 시 같은 조건 R2로 잡음 수준(|R1 − R2|)을 보고 원인을 조사한다. 원인 불명이면 S2 이후 중단.
- 이후 v4 dev 처리시간 비교 기준값 = R1 (v3 처리시간은 동시 실행으로 부풀어 비교하지 않는다).

## 1. 환경 (S0)

(아래 실행 후 기록)

| 항목 | 결과 |
|---|---|
| conda env | `pactory-seg` 신규 (python 3.12). n8n env의 torch는 건드리지 않음 |
| PyTorch | `torch==2.8.0+cu126`, `torchvision==0.23.0+cu126` (계획 §3 1순위로 성공, 대체안 불필요). `cuda.is_available()=True`, capability (6, 1), `get_arch_list()=[sm_61, sm_70, sm_75, sm_80, sm_86, sm_90]` (sm_61 포함), GPU 행렬곱 확인. 드라이버 560.94 |
| ultralytics | 8.3.253 (설치 후 torch 2.8.0+cu126 유지 확인). AGPL-3.0 |
| BlenderProc | 2.8.0 (pactory-seg). GPL-3.0 |
| Blender | 첫 `blenderproc run --blender-install-path C:\stockauto\tools\blender` 실행으로 설치: `C:\stockauto\tools\blender\blender-4.2.1-windows-x64` (Blender 4.2.1 LTS, 내장 Python 3.11.7). 탐침 스크립트 정상 종료 |
| 기타 (pactory-seg) | trimesh 5.2.0, opencv-python 5.0.0.93, numpy 2.5.4, imageio, PyYAML — 전체 `requirements_v4.txt` (pip freeze) |
| LM 모델 | HF bop-benchmark/lm `lm_models.zip` → `data/v4/bop/lm/models` (obj 1~15 + models_info.json, models_eval 포함). CC BY 4.0. `.gitignore`에 `data/v4/` 추가 |

## 2. v4.0 정리 항목

| 항목 | 처리 |
|---|---|
| 기준선 동결 | `run_v4 freeze-baseline` → `configs/v4/lmo_baseline.yaml` (= pipeline.yaml 기본 + profiles.lmo, `frozen: true`, v4 플래그 모두 꺼짐, 원본 pipeline.yaml 해시 기록). 테스트 01이 `pipeline_cfg("lmo")`와 같음을 검사 |
| other 비율 | `failures.other_failure_share()` (실패 건수 대비) 추가, `other_share_ok(basis=...)`, eval-bop·eval-dev summary에 `other_failure_share` |
| metadata label | `write_metadata(label=...)`. v3 `eval-bop`·`show-bop`과 v4 eval-dev는 "실측: BOP LM-O test_bop19 (합성 아님)". **기존 v3 결과 폴더(results/v3/eval-bop_*, show-bop_*)의 metadata label "합성 데이터…"는 오기**이며 폴더는 고치지 않았다 |
| 라이선스 | BOP 웹사이트(https://bop.felk.cvut.cz/datasets/, 2026-10-30 조회): LM-O CC BY-SA 4.0, LM CC BY 4.0 → `configs/v3/dataset.yaml` license에 기록. 공개 시 SA·AGPL(ultralytics) 고지 필요 |
| 스레드 | `run_v4`가 open3d import 전에 `OMP_NUM_THREADS`(없으면 16) 설정, `cv2.setNumThreads(1)`, metadata `runtime`과 `env.omp_num_threads`·`cv2_threads`에 기록 |

### 2.1 발견: Open3D 비결정성 (구현계획 §0-1 사실 정정)

1. **16스레드에서는 같은 프로세스·같은 시드에서도 RANSAC 결과가 달라진다.** 합성 하우징 장면에서 같은 호출 3회: 후보 fitness
   (1.0, 0.813, 0.983) / (1.0, 0.977, 1.0) / (0.997, 1.0, 0.980). 1스레드에서는 3회 모두 같다.
   → 계획의 "같은 스레드 수에서는 결정적"은 16스레드에서 성립하지 않는다. 16스레드 실행끼리의 차이는 잡음으로 다룬다(재현 B 기준).
2. **Open3D 전역 난수는 프로세스 시작 시 시드가 없다**(실행마다 다름, 1스레드에서도). v3 `eval-bop`은 첫 프레임의
   `segment_scene`(평면 RANSAC) 전에 시드를 주지 않으므로 v3 자신도 첫 프레임 분할이 실행마다 다르고, 그 차이가 다음 프레임 초기 난수 상태로 이어진다.
   (탐침: 같은 점군 `segment_plane`을 새 프로세스에서 3회 → 평면 계수 3가지; `o3d.utility.random.seed(5)` 후에는 같음.)
3. 대응: 기준선 경로는 v3와 같게 두었다(수정 없음). v4.1 경로는 프레임마다 분할 전에 `seed + im_id`로 재시드한다.
   재현 검증 A에는 `--o3d-init-seed`(시작 시 1회 시드, 기본 None = v3 동작)와 v3를 코드 수정 없이 같은 시드로 감싸 실행하는
   래퍼(scratchpad `v3_seeded.py`: `o3d.utility.random.seed(S)` 후 `vision_guide.run.main`)를 썼다.
   ~~TODO(Part B): eval-final의 기준선 실행에서도 시작 시드를 고정할지 결정해 동결 전에 기록~~ → 0.1b: eval-final은 시작 시 `--o3d-init-seed 0` (팀장 결정).

### 2.2 재현 검증 A (정확 일치, OMP_NUM_THREADS=1, 3프레임 23대상)

| 실행 | 결과 |
|---|---|
| A1: v3 `eval-bop --limit 3` vs v4 `eval-dev lmo_baseline --limit 3` (시작 시드 없음) | **불일치**: code 0건 다름, fitness 13·rmse 15·add 15건 다름. 프레임별로 보면 1·2번째(im 36, 47) 프레임만 다르고 3번째(im 61)는 7대상 모두 정확히 같음 → 원인은 위 2.1-2 (시작 난수 상태) |
| A2: 같은 비교, 양쪽 모두 시작 시 `o3d.utility.random.seed(0)` | **통과: 23대상 code·fitness·rmse_mm·add_mm 정확히 같음** (`results/v4/repro-A2_compare.json`) |

폴더: `results/v4/repro-A_v3_omp1`, `repro-A_v4_omp1`, `repro-A_compare.json`, `repro-A2_v3_omp1_seed0`, `repro-A2_v4_omp1_seed0`, `repro-A2_compare.json`.
판단: v4 기준선 코드 경로는 v3와 같은 계산이다 (A2). A1 불일치는 v3 자체의 비결정성이다.

### 2.3 재현 검증 B (전체 dev 366대상, OMP_NUM_THREADS=16, 단독 실행)

| 비교 | ADD(-S) Δ | 채택률 Δ | 오채택률 Δ | 대상별 (채택, 정답) 일치율 | 부트스트랩 CI ∋ 0 | 판정 |
|---|---:|---:|---:|---:|---|---|
| v3 기준 (`eval-bop_dbscan_20261029T233530`) vs **R1** | −0.27%p | −0.27%p | 0.00%p | **81.7%** (기준 ≥ 90%) | 예 (3지표) | **기준 미달** (일치율) |
| R1 vs R2 (같은 코드·설정·스레드, 잡음 진단) | −1.37%p | −0.55%p | +0.82%p | **80.3%** | 예 | 같은 수준의 불일치 |
| v3 기준 vs R2 | −1.64%p | −0.82%p | +0.82%p | 80.1% | 예 | 같은 수준 |

R1 (`results/v4/eval-dev_baseline_R1`): **ADD(-S) 32.2%, 채택률 86.9%, 오채택률 54.6%, 채택 정밀도 37.1%, 대상 영역 확보율 67.8%,
처리시간 중앙값 10.32 s / p95 15.45 s** (단독 16스레드). R2: ADD(-S) 30.9%, 오채택률 55.5%, 중앙값 10.60 s.

원인 판단 (S2 진행 근거):
- 집계 지표 기준(|Δ| ≤ 3%p, CI ∋ 0)은 모두 통과했고, **일치율 기준(≥ 90%)만 미달**했다.
- 같은 코드·같은 설정·같은 스레드 수로 두 번 돌린 R1 vs R2의 일치율도 80.3%로, v3 vs R1(81.7%)과 같은 수준이다.
  즉 대상별 불일치는 v4 코드 차이가 아니라 **실행 간 잡음**이다. 불일치는 50프레임 전체에 고르게 퍼져 있고(프레임당 0~4건),
  정답→오답 24건, 오답→정답 23건으로 대칭이다.
- 잡음 원인은 2.1의 두 가지(16스레드 RANSAC 비결정성, 시작 난수 무시드)로 확인됐고, 1스레드·시작 시드 고정에서는 정확히 일치한다(A2).
- 따라서 90% 일치율 기준은 "16스레드도 결정적"이라는 틀린 전제로 정한 값이었다. 기준을 바꾸지 않고 **미달로 기록**하며,
  원인이 확인됐으므로 S2 이후를 진행했다.
- 시사점: 같은 설정의 16스레드 dev 실행끼리도 ADD(-S)가 1~2%p, 대상별 결과가 약 20% 흔들린다. dev에서 1~2%p 차이는 잡음 범위다.
  처리시간 기준값은 R1 (중앙값 10.32 s).

## 3. v4.1 (S2, S3)

### 3.1 §7 항목

| 대상 | 검증 | 결과 | 근거 |
|---|---|---|---|
| 색 분포 | 단색 합성 물체: 같은 색 유사도 최대, 다른 색 낮음 | **통과**: 같은 색 > 0.99, 다른 색 < 0.2, 회색 < 0.05 (bhattacharyya·intersection). H 원형(175 vs 3 같은 구간), 무채색 V 구간, 시드 재현 | test_v4_02 |
| 깊이 일관성 | 정답 자세 위반 ≈ 0, 앞으로 옮긴 자세 위반 증가 | **합성 통과**: 정답 위반 < 1%, 3 cm 앞 > 90%, 가림물은 occluded, Raycaster == `cad.render_depth` (< 1e-5 m). **LM-O dev 실측: 사전 기준(중앙값 ≤ 0.05) 미달** — 첫 dev 프레임 8물체 위반 비율 0~0.20, 중앙값 0.064 (tol_free 15 mm). 위반은 실루엣 경계(내부 5px 침식 시 대부분 감소)와 Kinect depth의 +3~5 mm 편향, can(금속 반사)에서 큼. 기준을 바꾸지 않고 테스트를 `xfail(strict)`로 남겼다. 게이트 허용치(tol_free·max_violation)는 dev 격자로 정했다 | test_v4_04 |
| 픽셀 ↔ 점 대응 | 클러스터 점의 픽셀로 되돌린 RGB가 원래 위치와 일치 | **통과**: 역투영 → floor(투영) 픽셀 집합 정확히 같음, voxel 점 99% 이상 마스크 안, 좌표 부호화 RGB 회수 | test_v4_03 |
| 색 일치 | 유채색·무채색·원형 H·최소 픽셀·첫 실패 귀속 | **통과** | test_v4_05 |
| 실행부 | 플래그 끄면 기준선 == v3 호출 (1스레드), 2색 장면 색 순위, select_candidate 실시간 == 재생 벡터판 | **통과** | test_v4_06 |
| 통계 | 지표 정의, 부트스트랩 시드 결정성·동일 결과 CI ∋ 0·알려진 차이 검출·계수 쌍(H3) | **통과** | test_v4_11, test_v4_13 |

### 3.2 dev 결과 요약 (세부: `docs/v4/tuning_log.md`)

| | 기준선 R1 | v4.1 | gt_mask + 기준선 게이트 | gt_mask + v4.1 게이트 |
|---|---:|---:|---:|---:|
| ADD(-S) | 32.2% | **34.4%** | 58.5% | 60.9% |
| 오채택률 | 54.6% | **26.0%** | 13.4% | 5.7% |
| 정밀도 | 37.1% | **57.0%** | 81.4% | 91.4% |
| 처리시간 중앙값 | 10.32 s | **4.04 s** | 0.87 s | 0.93 s |
| 대상 영역 확보율 | 67.8% | 72.7% | — | — |

dev에서 v4.1은 오채택률을 절반 이하로 줄이고 처리시간을 약 60% 줄였다. ADD(-S) 차이(+2.2%p)는 CI가 0을 포함하고
실행 간 잡음(1~2%p) 수준이라 dev에서는 명확한 개선으로 보지 않는다. 게이트 값을 dev에서 골랐으므로 dev 수치는 낙관적이다.

## 4. 테스트

| 명령 | 결과 |
|---|---|
| `python -m pytest tests/v4 -q` | 36 통과, 1 xfail (test_v4_04 LM-O 실측 위반 기준, 위 3.1) |
| `python -m pytest tests -q` (전체 회귀, n8n) | **243 통과, 1 skip, 1 xfail** (기존 207 + v4 36), 72 s |
| `python -m pytest -q` (전체, n8n, 동결·매니페스트 후 2026-10-31) | **272 통과, 1 skip, 1 xfail**, 74 s. skip: test_v3_06 (bop_toolkit_lib 없음), xfail: test_v4_04 (3.1) |

## 5. 결과 폴더 (`results/v4/`)

| 폴더 | 내용 |
|---|---|
| `repro-A_*`, `repro-A2_*`, `repro-A*_compare.json` | 재현 A |
| `eval-dev_baseline_R1`, `eval-dev_baseline_R2`, `repro-B_compare_*.json` | 재현 B, 처리시간 기준값(R1) |
| `color-rank-diag` | 색 격자 144 설정 recall |
| `eval-dev_v41_dump_k20` (`candidates.csv`) | 후보 덤프 |
| `replay-tune` | 게이트 격자 3,840 조합 |
| `eval-dev_v41`, `eval-dev_v41_gt_mask` | v4.1 확인 실행, gt_mask 상한 |
| `seg-train_n` (`train_meta.json`, `val_synth.json`) | yolo11n-seg 합성 학습, 합성 val 지표·융합 conf |
| `seg-predict_dev` | dev 50프레임 YOLO-seg 예측 (conf ≥ 0.05) |
| `tune-v42_configs`, `eval-dev_v42_dump_e{0,2,4}` | 융합 격자 덤프 (erode 0/2/4, max_instances 3, compute-all) |
| `replay-v42` | 융합 격자 6 조합 재생·선택 |
| `eval-dev_v42` (`paired_bootstrap_dev.csv`, `compare_vs_*.json`, `seg_quality.csv`) | v4.2 단독 확인 실행 |

## 6. v4.2·동결 (Part B, S6~S7, 2026-10-31)

| 항목 | 결과 |
|---|---|
| 세그멘테이션 (합성 val_synth) | mask mAP50 0.988, mask mAP50-95 0.904, box mAP50-95 0.945 (`results/v4/seg-train_n/val_synth.json`). 융합 conf 0.55 (F1 곡선 최대 0.5576) |
| 세그멘테이션 (dev, 진단) | 대상 검출률(최고 IoU ≥ 0.5) 46.4%, conf 이상 마스크 없음 36.6%. 물체별 duck 97.5% ~ glue 10.0% (sim-to-real gap) |
| 융합 격자 (사전 등록 규칙) | erode_px 0, max_instances_per_class 1 (ADD − FA 0.232; tuning_log §3.4) |
| v4.2 dev 확인 실행 (단독) | ADD(-S) 29.2%, 오채택률 5.7%, 정밀도 83.6%, 처리시간 중앙값 0.69 s (tuning_log §3.5) |
| v4.2 − v4.1 (dev, 부트스트랩) | ADD(-S) −5.2%p [−11.0, +0.8] (0 포함), 오채택률 −20.2%p [−24.4, −16.1], 영역 확보율 −31.4%p [−37.8, −24.3] |
| 동결 | `lmo_baseline/v41/v42.yaml` 모두 `frozen: true`. `run_v4 freeze-manifest` → `configs/v4/frozen_manifest.json` (weights sha256 a1ca23be…d736, code_sha256 775a4164…26f1, combined 8afef09e…accf) |
| 회귀 | 4절 (272 통과, 1 skip, 1 xfail) |

dev 해석 (가설 판정 아님): v4.2는 채택한 것의 정밀도가 높고 오채택이 기준선의 약 1/10이지만, 합성 학습 분할기가 실측 dev에서
대상의 36.6%를 놓쳐 ADD(-S)는 v4.1보다 낮은 쪽이다(CI가 0을 겨우 포함). eval-final에서 H2는 지지되지 않을 가능성이 크다.
dev 결과를 보고 conf·격자·규칙을 바꾸지 않았다.

## 7. 남은 항목

- `eval-final` 1회 (단독 실행, 0.1b 조건) → `report_final.md`.
- 동결 후 `src/vision_guide/*.py`·설정·가중치를 바꾸면 매니페스트 검사로 eval-final이 거부된다.

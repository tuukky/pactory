# v3 검증 결과 (PLAN_v3 §7, 1~3주차 범위)

작성일: 2026-10-30 / 측정 PC: AMD Ryzen 7 3800X (8코어 16스레드), Windows 11, Python 3.12 (conda env `n8n`), CPU 전용.
라이브러리: open3d 0.19.0, opencv-python 4.12.0.88, trimesh 5.2.0, numpy 2.2.6, scipy 1.16.1 (`requirements_v3.txt`).

> LM-O 평가는 **dev 분할(50장)만** 했다. eval 분할은 어떤 명령으로도 평가·시각화하지 않았다.

## 1. 테스트

| 명령 | 결과 |
|---|---|
| `conda run -n n8n python -m pytest tests/v3 -q` | 45 통과, 1 skip (BOP toolkit 미설치 → 직접 대조 skip) |
| `conda run -n n8n python -m pytest -q` (전체 회귀) | **207 통과, 1 skip** (v1/v2 기존 162 + v3 45), 93 s (§8 수정 후) |

## 2. §7 항목별 결과

| 대상 | 검증 | 결과 | 근거 |
|---|---|---|---|
| 역투영·투영 | 왕복 오차 < 1e-9 | **통과**: 핀홀 max 4e-16 m, 왜곡 포함(cv2 반복 50회) < 1e-9 m. OpenCV projectPoints와 +0.5 규약 일치 | test_v3_01 |
| 역투영·투영 | 합성 Depth 역투영 점이 정답 메시 표면 위 | **통과**: 테스트 max < 2e-5 m (float32 레이캐스팅), synth-demo 3장면 15물체 max 1.2e-4 mm | test_v3_02, synth-demo `render_check.csv` |
| SE(3) | 합성·역변환·정규직교성 | **통과** | test_v3_01 |
| ICP | 알려진 변환 복원, 노이즈 0 오차 ≈ 0 | **통과**: 노이즈 0 te < 1e-6 m, re < 1e-4°, fitness 1.0. 노이즈 1 mm: te < 1 mm, re < 0.5° | test_v3_03 |
| FPFH+RANSAC | 최대 180° 초기 회전에서 ICP 수렴 범위(대칭 고려 15°, 1 cm) 진입 비율 | **측정**: 74% (50회, synth-demo; 0~60° 80%, 60~120° 69%, 120~180° 76%). 테스트 sanity 하한 ≥ 0.5 (12회 0.83) | test_v3_04, synth-demo `global_reg.csv` |
| 지표 | ADD/ADD-S를 BOP toolkit과 대조 | **통과(전사본 대조)**: bop_toolkit_lib `pose_error`의 add/adi/re 원문 전사본과 rel 1e-12 일치. 실제 패키지 대조 테스트는 미설치로 skip | test_v3_06 |
| Hand-eye | 노이즈 0에서 정답을 수치 오차 수준으로 복원 | **통과(조건부)**: PARK·HORAUD·ANDREFF·DANIILIDIS 각 250회 max 2.8e-10 mm / 1.3e-12°. **TSAI는 250회 중 1회(자세 5개) 퇴화**(1166 mm, 10.4°), 나머지 249회 수치 오차 수준 | test_v3_09, calib-handeye |
| Guidance | 손계산 장면 T_base_tcp 대조 | **통과**: 물체 base (0.65, 0.05, 0) 똑바로 → TCP (0.65, 0.05, 0.08), z_tcp = −z_base, 기울기 0 | test_v3_10 |
| 합성 데이터 | 렌더 Depth와 정답 자세 Point Cloud 일치 (샘플 장면) | **간이 장면만 통과** (위 표면 거리). BlenderProc 데이터는 4주차 | synth-demo `scene0_overlay.png` |
| v2 연동 | VP 100%·0초 → Rev.C와 동일 | **범위 밖** (이번 단계 제외) | — |

추가 검증: BOP 로더 단위·규약 — LM-O dev 2프레임 정답 자세 실루엣 vs `mask_visib` 일치율 > 0.99, depth 중앙값 차 < 15 mm
(test_v3_07). 합성 BOP 픽스처로 mm→m, depth_scale, eval 차단, 분할 결정성·불교차, BOP 대칭 규약, zip-slip 차단.

## 3. CLI 실행 결과 (`results/v3/`)

| CLI | 폴더 | 핵심 수치 |
|---|---|---|
| `show-bop` | `show-bop_20261029T232506` | dev scene 2 im 36, 288,976점, depth 0.862~1.731 m, PLY + PNG (Open3D 오프스크린 캡처 성공) |
| `eval-bop --split dev --seg dbscan` (주 결과) | `eval-bop_dbscan_20261029T233530` | 50장 366대상. **ADD(-S) 32.5%** (ADD 29.0%, ADI 37.4%), 채택 87.2% 중 오채택 54.6%p(채택했으나 오답). 객체별 4.4%(6 cat)~52.5%(11 glue). 분할 진단: 대상 클러스터(점 50% 이상이 정답 마스크)가 후보 20개 안에 있던 경우 66.9%, 그때 정답률 47.3%, 없을 때 2.5%. 처리시간 중앙값 10.6 s / p95 15.6 s (RANSAC 9.3 s = 후보 ~20개 × 재시작 5회) |
| `eval-bop --split dev --seg gt_mask` (정합 상한) | `eval-bop_gt_mask_20261029T233048` | 50장 366대상. **ADD(-S) 58.5%** (ADD 50.8%, ADI 70.2%, 게이트 무시 66.1%), 채택 71.9%. 처리시간 중앙값 870 ms / p95 1018 ms (RANSAC 813 ms가 지배) |
| `calib-intrinsics` | `calib-intrinsics_20261029T234519` | 20장 × 5회 × 밝기 노이즈 {0, 2, 5}. 재투영 RMS 0.058~0.071 px, \|Δfx\|·\|Δfy\| 중앙값 ≤ 0.34 px(최대 0.61 px, 0.1%), \|Δcx\|·\|Δcy\| ≤ 0.88 px, 투영 지도 오차 평균 0.27~0.66 px(중앙값) |
| `calib-handeye` | `calib-handeye_20261029T234254` | 노이즈 4수준 × 자세 {5,10,15,20,30} × 5방법 × 50회. 노이즈 0: 위 §2. 기준 조건(σ 0.05°/0.25 mm, 15자세) PARK 중앙값 0.85 mm / 0.064°. 최고 노이즈(0.1°/0.5 mm) 30자세 0.96~1.6 mm / 0.087~0.095°. 종단(렌더링+solvePnP, 15자세, 밝기 σ2, 5회): PnP 0.026°/0.066 mm, Hand-eye 중앙값 0.25~0.50 mm / 0.015~0.019° |
| `guidance-budget` | `guidance-budget_20261029T235236` (자세 σ config), `..._235316` (synth-demo 경험 σ) | 아래 §4 |
| `synth-demo --n 3` | `synth-demo_20261029T235040` | 3장면 × 5물체, 노이즈 1 mm, 결측 5%. 클러스터 15, 채택 13 (오채택 0), 채택 중 파지 성공 84.6%. ADD-S 정답인데 파지 실패 2건(1건은 뒤집힘 180° 오인식, TCP 오차 80 mm). 장면당 4.4 s, 클러스터당 0.91 s(중앙값). 경험 자세 σ(뒤집힘 제외 12건): 축 0.14°, 병진 0.58 mm |

반복 수 감축: 없음 (calib-intrinsics 기본 5회, calib-handeye 50회·종단 5회, guidance-budget 표본 4000·Hand-eye σ 추정 200회).

처리시간 주의: gt_mask 평가 중 v3 pytest(캘리브레이션 테스트)가, dbscan 평가 중 calib-handeye·calib-intrinsics·synth-demo·
guidance-budget·전체 pytest가 동시에 실행됐다. RANSAC이 멀티스레드라 처리시간이 수~수십 % 부풀었을 수 있다. 최종 처리시간 분포는
4주차 평가 세트에서 단독 실행으로 다시 잰다.

비결정성: Open3D RANSAC은 멀티스레드라 시드를 고정해도 결과가 미세하게 달라진다. synth-demo 같은 시드 2회 실행에서 채택 15 → 13,
전역 정합 수렴 비율 0.68 → 0.74가 나왔다(첫 실행은 요약 정의 변경으로 `_superseded/`로 옮김).

## 4. 오차 예산 (§6.4)

기준 장면: 빈 중심 똑바로 놓인 하우징, 윗면 파지, 카메라-물체 1.0 m. 허용치 ±2 mm / ±3°, 성공률 기준 95% (gripper.yaml 사전 기재).

| 요인 (축별 σ) | 성공률 | 위치 오차 중앙값 / p95 | 평균제곱 비중 |
|---|---|---|---|
| 자세만 (0.3°, 0.5 mm, 합성 가정) | 98.6% | 0.93 / 1.71 mm | 35% |
| Hand-eye만 (0.042°, 0.59 mm; 기준 조건 PARK 200회 RMS/√3) | 88.5% | 1.21 / 2.32 mm | 64% |
| 합성 | 72.6% | 1.54 / 2.88 mm | 100% |

- **주 결과**: "기준 Hand-eye 조건(측정 노이즈 σ 0.05°/0.25 mm, 로봇 자세 15개)의 병진 오차와 자세 오차만으로도
  TCP 허용치 ±2 mm를 95% 이상 지키지 못한다 (Hand-eye 회전 σ 0°에서도 98.6% → 합성 시 미달)." Hand-eye 병진 오차가 가장 큰 요인이다.
- **보조(사후 추가, Hand-eye 병진 σ = 0 가정)**: "TCP 위치 허용치 ±2 mm를 95% 이상 지키려면 Hand-eye 회전 오차가
  축별 σ 0.025° (회전각 중앙값 0.039°) 이하여야 한다." 카메라-물체 1.0 m에서 0.039°는 약 0.7 mm 위치 오차에 해당한다.
- 자세 σ를 synth-demo 경험값(0.14°/0.58 mm)으로 바꿔도 같은 결론(합성 73.5%, 보조 한계 0.025°).
- 접근축 오차는 p95 0.73° 이하로 허용치 3°에 여유가 크다. 실패는 모두 위치 오차 때문이다.

## 5. 임계값·설계 조정 기록 (dev 결과로만)

| # | 항목 | 변경 | 근거 (dev) |
|---|---|---|---|
| 1 | `pipeline.ransac.n_restarts` | 1 → 5 | 간이 합성 장면 16개(시드 0~7 × 노이즈 0/2 mm): 채택·정답 29 → 65 / ~104 클러스터, 오채택 0 유지. 장면당 1.1 → 5.0 s |
| 2 | 재시작 간 선택 기준 | ICP fitness → fitness × 가시 비율 | 같은 실험. 위에서 본 평면 원판(플랜지 윗면)이 잘못된 면에 fitness 1.0으로 맞는 사례를 가시 비율로 거름 |
| 3 | `profiles.lmo.plane.distance_m` | 0.006 → 0.015 (LM-O만) | dev 첫 6프레임 44대상: Kinect 테이블 depth가 6 mm 띠보다 두꺼워 평면이 두 장으로 갈라짐(계수 거의 같은 평면 1·2위). 대상이 크기 통과 클러스터에 존재 26 → 35 |
| 4 | `profiles.lmo.plane.min_inlier_frac` | 0.15 → 0.10 (LM-O만) | 같은 진단: 테이블 비율 0.136~0.152 프레임에서 평면 미제거 → 대상이 테이블과 합쳐짐(bbox 9~13×직경) 9 → 5건 |
| 5 | `profiles.lmo.cluster.max_candidates` | 10 → 20 (LM-O만) | 대상 클러스터가 가까운 순 10위 안 14~17 / 44, 20위 안 19~27 / 44 |
| 6 | BOP 내부 파라미터 규약 | OpenCV(+0.5) → BOP K 그대로 | dev 첫 프레임 실루엣 일치율 0.95~0.99 → 0.999 (model_spec §2) |
| 7 | Hand-eye 전처리 | base를 첫 로봇 자세 방향으로 고정 회전 후 풀고 복원 (전 방법) | 전처리 없이 TSAI가 노이즈 0에서 180° 퇴화, 노이즈가 있어도 자세가 늘수록 오차 증가(0.1°/0.5 mm·30자세 중앙값 39 mm). 동치 변환이며 PARK는 전처리 유무로 1e-9 이내 동일. 전처리 전 결과 `_superseded/` 보관 |
| 8 | test_v3_01 왜곡 왕복 테스트 점 | 무작위 → 시야(영상) 안 점 | 첫 작성 시 점이 영상 밖(정규화 좌표 1.33)까지 나가 왜곡 모델이 비단조 → 2.7e-4 m. 테스트 설계 오류이며 허용 오차(1e-9)는 바꾸지 않음 |
| 9 | guidance-budget 보조 분석 | Hand-eye 병진 σ=0 회전 한계 추가 | 주 결과(사전 설정)가 "격자 최소값에서도 미달"이라 §6.4 질문의 회전 한계를 별도로 보임. 사후 추가로 표시 |

LM-O 조정(3~5)은 dev 첫 6프레임 진단으로만 정했고, 정한 뒤 dev 50장을 1회 평가했다. 평가 결과를 보고 다시 바꾸지 않았다.
`pipeline.yaml`은 아직 `frozen: false`이며 4주차 BlenderProc 개발 세트 조정 후 동결한다.

## 6. 알려진 문제

- LM-O dbscan(전체 파이프라인): 색 정보 없이 기하만으로 대상 객체를 고르므로 다른 물체·배경 클러스터에 높은 점수로 오채택하는
  경우가 많다(오채택 54.6%p, 실패 코드는 LOW_VISIBILITY 12.6%, LOW_FITNESS 0.3%뿐이라 신뢰도 게이트가 오답을 거의 못 거른다). 정합 상한(gt_mask)과의 차이가 분할·식별 단계의 손실이다.
- LM-O 실측 depth는 정답보다 수 mm 멀다(Kinect 편향). 보정하지 않아 병진 오차에 포함된다.
- ADD-S는 하우징 뒤집힘(180°)을 정답으로 판정할 수 있다(ADI 3~4 mm). 파지 성공 판정(TCP 허용치)이 이를 잡는다 → 4주차 실패 분석에서 축 오차를 함께 보고해야 한다.
- 가시 비율은 정답 자세여도 가림이 크면 낮아 LM-O gt_mask에서 정답 자세 일부를 거부한다(게이트 무시 66.1% vs 58.5%).
- 체커보드 렌더링 앨리어싱(4×4 서브샘플)과 k3 상관 때문에 노이즈 0에서도 fx 오차 ~0.3 px, 투영 지도 오차 ~0.66 px가 남는다.
- OpenCV TSAI는 전처리 후에도 노이즈 0·자세 5개에서 250회 중 1회 퇴화했다.
- 코너 순서 180° 모호성은 보드 roll ±20° 가정으로 풀었다(TODO: 실제 보드는 ChArUco).

## 8. 디버깅 수정 (2026-10-30, W1~W4)

### W1 — dev/eval 인접 프레임 누설 (eval 재생성)

- 문제: 최초 eval(`make_split`)이 LM-O scene 2 연속 시퀀스에서 dev와 im_id 1~2 차이로 붙은 쌍을 가졌다 (최소 간격 1).
- 조치: **eval은 한 번도 평가·로드하지 않은 상태**에서 재생성. dev 50장은 임계값 조정에 이미 쓰였으므로 그대로 두었다.
  규칙: test_targets_bop19 200장 중 dev가 아니고 같은 scene의 모든 dev 프레임과 |im_id 차| > 5 (`eval_gap_im: 5`)인 후보 91장에서
  `split_seed`로 50장 (`datasets.make_gapped_eval`). 목록 계산은 targets json의 id만 썼다(프레임 파일 미접근).
- 결과: 새 eval 50장, **dev-eval 최소 간격 6 im_id**. `dataset.yaml`에 규칙·사유 주석 기록.
- 테스트: `test_lmo_split_matches_dataset_yaml`(dev = make_split dev, eval = make_gapped_eval), `test_lmo_dev_eval_min_gap`,
  `test_make_gapped_eval_deterministic_and_gapped`.
- INFO2: `load_frame(forbid_ids=...)` — 금지 id면 파일을 열기 전에 ValueError. `iter_targets(split="dev")`와 `show-bop --scene --im`이
  eval id 집합을 넘긴다 (`test_load_frame_forbid_ids`).

### W2 — 요약 통계 정의

- `metrics.summarize`: `median/p95_te_mm`, `median/p95_re_deg`는 **채택(code == OK) 추정만**, 회전은 유효 회전 오차(대칭 물체
  `re_sym_deg`, 그 외 `re_deg`). 추정 자세가 있는 전체 대상 기준은 `raw_median/raw_p95_*` 열로 별도.
- 재평가 없이 `python -m vision_guide.run resummarize <폴더>`로 기존 per_target.csv에서 요약만 다시 계산했다. 이전 요약은
  `summary_prev.json`, `summary_by_object_prev.csv`로 보존. 성공률·채택률은 정의가 같아 변하지 않았다.

| 폴더 (ALL 행) | median te (mm) | p95 te (mm) | median re (°) | p95 re (°) |
|---|---|---|---|---|
| dbscan 이전 (전체 추정, re 비대칭식) | 307.3 | 950.2 | 106.3 | 177.5 |
| dbscan 이후 (채택만, 유효 re) | 278.2 | 957.3 | 90.8 | 176.0 |
| dbscan 이후 raw_ (전체 추정, 유효 re) | 307.3 | 950.2 | 93.2 | 176.1 |
| gt_mask 이전 | 8.0 | 49.2 | 6.4 | 178.9 |
| gt_mask 이후 (채택만, 유효 re) | 6.0 | 24.3 | 3.0 | 175.5 |
| gt_mask 이후 raw_ | 8.0 | 49.2 | 5.1 | 176.8 |

p95 re가 여전히 175° 이상인 것은 비대칭 물체의 뒤집힘(180°) 오채택 때문이다(§6).

### W3 — Hand-eye 실패 처리

- `handeye_reference_sigma`: RMS를 `np.nanmean`으로 계산하고 `n_failed`(cv2 실패 시행 수)를 info에 기록. 실패 > 0이면 info `warning`,
  `RuntimeWarning`, `guidance-budget` stderr 경고.
- `handeye_end_to_end`: `calibrate_handeye` 호출을 `handeye_curve`와 같이 `cv2.error` → nan 처리.
- 기존 결과(`calib-handeye_*`, `guidance-budget_*`)는 실패가 없었던 조건이라 재실행하지 않았다.

### 기타

- INFO3: `eval-bop`에서 대상 객체가 scene_gt에 없으면 예외로 죽지 않고 `OTHER`로 귀속한다.
- INFO8: `guidance-budget --pose-sigma-from`의 summary에 `pose_sigma_empirical`이 None이면 TypeError 대신 명확한 메시지로 종료.

### W4 — BOP cam_K 픽셀 중심 규약 (코드 변경 없음)

BOP cam_K는 렌더러 `mask_visib` 일치 기준(일치율 0.999 vs +0.5 적용 시 0.95~0.99)으로 **+0.5를 적용하지 않는다**.
bop_toolkit의 역투영 규약과는 0.5 px 차이가 날 수 있다(1 m에서 약 0.9 mm). 성공 임계 0.1×직경(LM-O 최소 직경 약 100 mm → 10 mm)
대비 미미하다.


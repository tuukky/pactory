# 프로젝트 개요와 단계별 범위

pactory는 4단계로 진행했다. 앞 단계를 동결한 뒤 다음 단계를 시작했고, 단계마다 계획서(PLAN)를 먼저 쓰고 판정 기준과 가설을 실행 전에 고정했다.
계획서 원본은 저장소에 포함하지 않았다. 코드와 문서에 나오는 `PLAN §n`, `PLAN_v2 §n` 표기는 그 계획서의 절 번호다. 각 절에서 정한 내용은 단계별 `model_spec.md`, `verification.md`에 옮겨 두었다.

| 단계 | 패키지 | 주제 | 상태 |
|---|---|---|---|
| v1 | `line_sim` | 3품종 공유설비 라인의 이산사건 시뮬레이션 (엔진·검증) | 엔진·검증 완료, 개선안 비교는 v2로 넘김 |
| v2 | `factory_sim` | 모터 공장 공정설계 (Rev.A → Rev.B → Rev.C) | 완료·동결 |
| v3 | `vision_guide` | RGB-D 6D 자세 추정, 캘리브레이션, 로봇 가이던스 기준선 | 기준선 완료·동결 |
| v4 | `vision_guide` | 자세 추정 개선 (색 기반 식별, 합성 데이터 YOLO-seg 융합) | eval 1회 평가 완료 |

## v1 — 생산라인 시뮬레이터

- 대상: 가상 소형 모터 3품종(P1·P2·P3), 8시간 1교대. 입력값은 모두 합성 가정(`data/assumptions.csv`).
- 구현: SimPy 기반 이산사건 엔진. 유한 버퍼와 막힘, 병렬 조립 2대와 품종 전환 준비시간, 검사·1회 재작업.
- 검증: A→B→C 직렬 라인 해석해(1시간 완료 수 119개), 물량·상태시간·WIP 면적 보존, 동일 시드 재현 (`docs/verification.md`).
- 개선안 비교 실험은 discovery 단계까지만 돌렸고, 공정설계는 v2의 공장 모델로 확장해 진행했다.

## v2 — 모터 공장 공정설계

- 설비 21대, 로터·스테이터 서브라인, BOM 키트 합류, 작업자 8명(숙련 매트릭스), AGV 3대, 하우징 (s, Q) 재고, 설비 고장.
- 절차: 정적 부하 계산 → 손실 분해(7범주) → 단일 변경 후보 비교 → 개정 동결 → 독립 확인 시드로 재확인.
- 사전 가설과 판정:
  - H1 (1차 병목 = 정적 부하가 가장 큰 공정): **기각**. 정적 계산은 오븐을 지목했지만 실제 제약은 하우징 재고 정책과 투입 통제였다.
  - H2 (병목을 해소하면 다른 자원 유형으로 이동): **지지**. 자재 정책 → 투입 통제·오븐 → AGV·포장 작업자 순으로 이동했다.
- 결과: Rev.A(수요 충족 0.741) → Rev.B(하우징 재주문점 s 300→800, 0.992) → Rev.C(+키트 슬롯 30→45, 수요 +15%에서 0.993). 설비·인원 추가 없이 운영 규칙만 바꿨다.
- 상세: `docs/v2/final_report.md`.

## v3 — 비전 가이드 기준선

- 학습 없이 Depth 형태 정보만 쓰는 파이프라인: 평면 제거 → DBSCAN 분할 → FPFH+RANSAC → ICP → 신뢰도 검사 → TCP 목표 계산.
- 데이터: BOP LM-O(실측, 가림이 심한 다물체 장면). dev 50장으로만 조정하고 eval 50장은 v4 최종 평가 전까지 쓰지 않았다.
- dev 결과: ADD(-S) 32.5%, 정답 영역을 주면 58.5%. 손실 대부분이 정합이 아니라 분할과 대상 식별에서 생겼고, 채택 결과의 63%가 오답이었다.
- 캘리브레이션: 체커보드 내부 파라미터, eye-to-hand Hand-eye 5개 방법 비교. 오차 예산상 Hand-eye 병진 오차가 TCP 허용치(±2 mm)의 가장 큰 위협.
- 계획했던 v2 라인 영향 실험(비전 피킹 스테이션)은 범위에서 제외했다.
- 상세: `docs/v3/report_baseline.md`, `docs/v3/verification.md`.

## v4 — 자세 추정 개선

- v4.1: 물체 모델의 정점 색으로 후보 순위를 매기고, 깊이 일관성(자유공간 위반)과 색 일치도를 신뢰도 검사에 추가.
- v4.2: BlenderProc 합성 데이터로 YOLO11n-seg를 학습해 분할을 대체하고, 그 마스크 영역에 v4.1 정합을 적용. LM-O test 영상은 학습에 쓰지 않았다.
- 세 버전(baseline, v4.1, v4.2)을 동결하고 해시를 기록한 뒤(`configs/v4/frozen_manifest.json`) eval 50장에서 한 번만 평가했다.

| 버전 | ADD(-S) | 오채택률 | 채택 정밀도 | 처리시간 중앙값 |
|---|---:|---:|---:|---:|
| baseline | 28.8% | 62.7% | 31.5% | 10.2 s |
| v4.1 | 33.1% | 27.7% | 54.4% | 4.0 s |
| v4.2 | 32.2% | 8.8% | 78.6% | 0.8 s |

- 가설 판정: 성공률 향상(H1, H2)은 명확한 차이를 확인하지 못했다. 오채택률 절반 이하(H3)와 처리시간 단축(H4)은 지지됐다.
- 즉 개선의 효과는 "더 많이 맞히기"보다 "틀린 결과를 덜 내보내기"와 속도에 있었다. 로봇 피킹에서는 오채택이 더 위험한 실패다.
- 상세: `docs/v4/tuning_log.md`, `docs/v4/verification.md`.

## 설치

Python 3.11 이상 (개발 환경 3.12–3.13, Windows 11).

```powershell
python -m pip install -e ".[dashboard,dev]"     # v1·v2 엔진, 대시보드, pytest
python -m pip install -r requirements_v3.txt    # v3·v4 CPU 파이프라인 (open3d, opencv 등)
```

v4.2의 세그멘테이션 학습·추론과 BlenderProc 렌더링은 별도 환경(CUDA PyTorch, ultralytics, blenderproc)에서 실행한다. 패키지 목록은 `requirements_v4.txt`에 있다.

- 세그멘테이션용 Python 경로는 환경변수 `PACTORY_SEG_PYTHON`으로 지정한다.
- BlenderProc 경로는 `configs/v4/train_synth.yaml`의 `tools` 항목에 있다. 이 파일은 동결 해시(`configs/v4/frozen_manifest.json`)에 포함되므로, 경로를 바꾸면 eval-final 사전 검사가 실행을 거부한다.

```powershell
python -m pytest tests -q
```

## 데이터

BOP 데이터셋과 합성 데이터, 학습 가중치는 저장소에 포함하지 않는다.

```powershell
python -m vision_guide.run download-bop    # LM-O (CC BY-SA 4.0) → data/v3/bop/lmo
python -c "from pathlib import Path; from vision_guide.datasets import download_bop; download_bop('lm', Path('data/v4/bop'), parts=('models',))"    # v4 합성 방해물용 LM 모델 (CC BY 4.0)
```

합성 학습 데이터는 `python -m vision_guide.run_v4 synth-gen --split train_synth`로 다시 만든다.

## 실행

### v1 — 3품종 라인

```powershell
python -m line_sim.run --config configs/baseline.yaml --seed 100 --output results/demo
python -m line_sim.experiments --matrix configs/experiments.yaml --phase discovery
python -m streamlit run app/dashboard.py
```

### v2 — 모터 공장

```powershell
python -m factory_sim.static_capacity --config configs/v2/rev_A.yaml          # 정적 부하 계산
python -m factory_sim.run --config configs/v2/rev_C.yaml --seed 100 --output results/v2/demo_c --detail full
python -m factory_sim.calibration --config configs/v2/rev_A.yaml --seeds 1-9 --workers 12
python -m factory_sim.experiments run --study A_diag --phase explore --workers 12
python -m factory_sim.experiments run --study rev1 --phase explore --workers 12
python -m factory_sim.experiments freeze --study rev1 --candidate H1 --as B
python -m factory_sim.experiments run --study final --phase confirm --workers 12
python -m factory_sim.experiments run --study p2_final --phase confirm2 --workers 12
python -m streamlit run app/dashboard_v2.py
```

- 실험 결과: `results/v2/<study>_<phase>_<UTC>_<hash8>/`. full 상세 run은 run당 55–90MB라 실험은 summary 모드로 저장한다.
- Windows 병렬 실험은 spawn 방식이며 `--workers`로 프로세스 수를 정한다.

### v3 — 비전 가이드 기준선

```powershell
python -m vision_guide.run eval-bop --split dev --seg dbscan
python -m vision_guide.run calib-intrinsics
python -m vision_guide.run calib-handeye
python -m vision_guide.run guidance-budget
```

### v4 — 자세 추정 개선

```powershell
python -m vision_guide.run_v4 eval-dev --config configs/v4/lmo_v41.yaml
python -m vision_guide.run_v4 eval-dev --config configs/v4/lmo_v42.yaml
python -m vision_guide.run_v4 compare-runs <ref_dir> <new_dir>
python -m vision_guide.run_v4 eval-final --confirm-eval-once    # 1회만 실행 가능 (잠금 파일)
```

- Open3D RANSAC 결과는 스레드 수에 따라 달라진다. `run_v4`는 open3d를 import하기 전에 `OMP_NUM_THREADS`를 정하고(기본 16), 실제 값을 metadata.json에 기록한다.
- eval 분할은 `eval-final` 안에서만 읽는다.

### v5 — 비전 → 파지 → 라인 통합

```powershell
python -m vision_guide.bin_v5 render --split dev|eval|train|val          # BlenderProc 하우징 bin (pactory-seg env)
python -m vision_guide.seg_bin_v5 train | val | predict --split eval     # B1 학습·예측 (WSL2 또는 CUDA env)
python -m vision_guide.bin_v5 eval-final --pipeline B0,B1 --preds results/v5/s1_bin/seg-predict_eval
python tools/v5/s1_judge.py results/v5/s1_bin/eval-final_<ts>
# S2: 별도 mujo 프로젝트 (WSL2, 코드·실험 버전 미포함) — python tools/run_housing_pick.py --records <S1 pick_records.jsonl> --split eval --lock
python tools/v5/build_vp_dist.py --s1 <S1> --s2 <S2> --condition base --out configs/v5/vp_dist_measured.json
python tools/v5/run_line_vp.py --phase confirm                            # 1회만 (잠금 파일)
```

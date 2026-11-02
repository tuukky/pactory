# pactory — 생산라인 시뮬레이션과 비전 가이드 피킹

가상 모터 공장을 대상으로 한 두 갈래의 작업이다.

- **공정설계 (v1–v2)**: SimPy 이산사건 시뮬레이션으로 병목을 진단하고, 초기 설계(Rev.A)를 개정해 수요와 납기를 맞추는 공정 구성을 찾는다.
- **비전 가이드 피킹 (v3–v4)**: RGB-D로 물체의 6D 자세를 추정해 로봇 파지 위치를 계산하고, 공개 데이터셋(BOP LM-O)에서 실패 원인을 분석해 개선한다.

공장 모델의 입력값(작업시간·불량률·고장 등)은 모두 합성 가정이다. 비전 평가는 실측 공개 데이터셋 LM-O를 썼고, 학습 데이터는 BlenderProc 합성 렌더링이다. 한계는 각 단계의 `limitations.md`에 정리했다.

단계별 범위와 결과 요약: [docs/project_overview.md](docs/project_overview.md)

## 주요 결과

### v2 공정설계 — 운영 규칙 두 개로 수요 +15% 대응

| 설계 | 변경 (Rev.A 대비) | 수요 | 수요 충족률 | 납기 충족률 | 리드타임 |
|---|---|---|---:|---:|---:|
| Rev.A | — | 18대/h | 0.741 | 0.423 | 35.8h |
| Rev.B | 하우징 재주문점 s 300→800 | 18대/h | 0.992 | 0.994 | 2.0h |
| Rev.B | | 20.7대/h | 0.964 | 0.993 | 7.0h |
| Rev.C | + 키트 슬롯 K 30→45 | 20.7대/h | 0.993 | 0.993 | 2.0h |

- 정적 부하 계산은 오븐을 병목으로 꼽았지만, 실제로 처리량을 제한한 것은 하우징 재고 정책과 투입 통제였다.
- 설비·인원·AGV를 늘리지 않고 운영 규칙 두 개만 바꿨다. 확인 실험은 탐색에 쓰지 않은 독립 시드 30개로 했다.
- 상세: [docs/v2/final_report.md](docs/v2/final_report.md)

### v4 자세 추정 — 오채택률 62.7% → 8.8%

LM-O eval 50장, 세 버전을 동결한 뒤 1회 평가.

| 버전 | 방법 | ADD(-S) | 오채택률 | 채택 정밀도 | 처리시간 중앙값 |
|---|---|---:|---:|---:|---:|
| baseline (v3) | Depth 형태만: DBSCAN + FPFH/RANSAC + ICP | 28.8% | 62.7% | 31.5% | 10.2 s |
| v4.1 | + 색 기반 후보 순위, 깊이 일관성 검사 | 33.1% | 27.7% | 54.4% | 4.0 s |
| v4.2 | 합성 데이터로 학습한 YOLO11n-seg + 기하 정합 | 32.2% | 8.8% | 78.6% | 0.8 s |

- 성공률 향상은 통계적으로 명확하지 않았다. 효과는 틀린 결과를 덜 내보내는 쪽(오채택 감소)과 속도에서 나왔다.
- 상세: [docs/v4/tuning_log.md](docs/v4/tuning_log.md), [docs/v4/verification.md](docs/v4/verification.md)

## 개발 방식

구현에는 AI 코딩 도구(Claude Code)를 활용했다. 문제 정의, 공정 구조와 가정, 실험 설계와 판정 기준, 결과 해석과 다음 단계 결정은 직접 했다. 단계마다 판정 기준과 가설을 실행 전에 고정하고, 결과가 가설과 다르면 다르다고 기록했다.

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
python -m pytest tests -q    # 전체 272 통과, 1 skip, 1 xfail
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

## 파일 형식

- 시간 단위는 초. UI에서만 분·시간으로 바꾼다.
- CSV는 UTF-8 with BOM(`utf-8-sig`)으로 저장한다. Excel에서 한글이 깨지지 않게 하기 위해서다.
- 모든 실행은 결과 폴더에 `config_resolved.yaml`(또는 설정 해시)과 `metadata.json`(패키지 버전, git 해시, 시드)을 남긴다.

## 폴더 구조

```text
pactory/
├── src/
│   ├── line_sim/       v1 이산사건 엔진
│   ├── factory_sim/    v2 공장 엔진 (AGV, 작업자, 재고, 고장, 배차)
│   └── vision_guide/   v3·v4 자세 추정, 캘리브레이션, 가이던스, 합성 데이터, 세그멘테이션
├── configs/            v1 baseline·experiments, v2 Rev.A–C, v3 pipeline, v4 동결 설정
├── data/               합성 가정 CSV (데이터셋은 download-bop으로 받음)
├── app/                Streamlit 대시보드 (저장된 결과만 읽음)
├── tests/              v1–v4 검증 테스트
└── docs/               단계별 model_spec, verification, 보고서, limitations
```

`line_sim/logging.py`는 표준 라이브러리 `logging`과 이름이 같다. 패키지 안에서는 상대 import만 쓰므로 충돌하지 않지만, 모듈 파일을 직접 실행하면 표준 모듈이 가려질 수 있으니 `python -m line_sim.<모듈>` 형태로 실행한다.

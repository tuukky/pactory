# pactory — 생산라인 시뮬레이션과 로봇 피킹

가상 모터 공장에서 주문이 밀리는 원인을 찾고, 카메라로 부품을 찾아 집는 공정을 추가했을 때 생산량이 어떻게 바뀌는지 실험했습니다. 생산라인은 SimPy, 부품 자세 추정은 RGB-D, 로봇 파지는 MuJoCo로 구현했습니다.

| 실험 | 결과 |
|---|---|
| 모터 생산라인 | 하우징 발주점과 키트 대기 슬롯을 조정해 시간당 20.7대 수요에서 수요·납기 충족률을 각각 99.3%로 맞췄습니다. 확인 실험은 독립 시드 30개로 진행했습니다. |
| 부품 자세 추정 | LM-O 평가 이미지 50장에서 오채택률이 62.7%에서 8.8%로 줄었고, 처리시간 중앙값은 10.2초에서 0.8초로 줄었습니다. 자세 추정 성공률의 개선은 이 평가에서 뚜렷하지 않았습니다. |
| 비전과 파지 통합 | 합성 장면에서 채택된 자세 22건은 MuJoCo에서 모두 집었고, 위아래가 뒤집힌 자세는 7건 모두 실패했습니다. 이 결과로 만든 분포를 라인에 적용했을 때 수요 충족률은 99.5%였습니다. 기존 Rev.C와 차이는 확인되지 않았습니다. |

공장 입력값과 v5 파지 장면은 합성입니다. v4 자세 추정에는 공개 RGB-D 데이터셋 LM-O를 사용했습니다. 실물 공장이나 로봇에서의 검증은 범위에 포함되지 않습니다.

## 실행

Python 3.11 이상에서 모터 라인 예제를 실행합니다.

```powershell
python -m pip install -e ".[dashboard]"
python -m factory_sim.run --config configs/v2/rev_C.yaml --seed 100 --output results/v2/representative/rev_C --detail full
python -m streamlit run app/dashboard_v2.py
```

v5 라인 실험은 저장소에 포함된 파지 분포로 실행할 수 있습니다.

```powershell
python tools/v5/run_line_vp.py --phase dev
```

v5 전체를 처음부터 재현하려면 BlenderProc, 학습 데이터와 가중치, 별도 `mujo` 프로젝트의 `tools/run_housing_pick.py`가 필요합니다. 이 저장소에 `mujo` 코드와 해당 실험에 사용한 버전은 포함되어 있지 않습니다.

- [단계별 구성과 상세 실행 방법](docs/project_overview.md)
- [생산라인 실험](docs/v2/final_report.md) · [자세 추정 검증](docs/v4/verification.md) · [비전·파지·라인 통합 보고서](docs/v5/final_report.md)

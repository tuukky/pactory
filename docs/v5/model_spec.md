# v5 모델 명세 — 비전 → 로봇 파지 → 공장 라인 통합

v5는 v3·v4(비전)와 v2(공장)를 MuJoCo 로봇 팔 프로젝트(`C:\stockauto\code\mujo`)로 이어 붙인다.
계획·가설·판정 규칙은 `PLAN_v5.md`(저장소 제외)에 결과보다 먼저 고정했다. 여기서는 구현이 실제로 무엇을 하는지 정리한다.
모든 입력은 합성 데이터와 합성 가정이다.

## 1. 흐름과 파일 계약

```
S1 비전 (Windows, conda n8n / pactory-seg)           S2 파지 (WSL2, MuJoCo 3.17)            S3 라인 (factory_sim)
BlenderProc 하우징 bin → B0/B1 → 집기 결정   ──►   추정 자세로 접촉 파지 → 결과   ──►   Rev.C + VP 셀
            results/v5/s1_bin/*/pick_records.jsonl        mujo/results/housing_pick/*/pick_result.jsonl
                                       └──────── tools/v5/build_vp_dist.py → configs/v5/vp_dist_measured*.json ─┘
```

두 실행 환경은 JSONL 파일로만 주고받는다. 각 결과 폴더의 `metadata.json`에 입력 파일 해시와 소스 해시를 남긴다.

## 2. S1 — 하우징 bin 비전

| 항목 | 구현 |
|---|---|
| 장면 | `bproc_bin_scene.py` (Blender 안). 빈 0.60 × 0.40 m, 벽 0.12 m. 하우징 N개를 무작위 자세로 떨어뜨려 물리 안정화 (볼록껍질 충돌) |
| 카메라 | `configs/v3/camera.yaml` 고정 eye-to-hand (빈 위 1.0 m). 학습 장면만 위치 ±6 cm, 시선 ±5 cm 흔들림 × 10뷰 |
| 조건 | 기준(적재 5, σ 1 mm, 결측 5 %, 기본 조명)에서 한 변수씩 11조건 (`configs/v5/dataset_bin.yaml`). 노이즈·결측은 같은 렌더에 후처리 |
| 결측 모델 | 하우징 픽셀만, 저주파 잡음장(가우시안 σ 6 px) 상위 p 분위 → 덩어리 모양 (반사 모사) |
| B0 | v3 기준선: 빈 내부 ROI → DBSCAN → FPFH+RANSAC → ICP → 신뢰도 (`pipeline_bin_B0.yaml`) |
| B1 | 하우징 전용 YOLO11n-seg 마스크(위쪽부터) → 마스크 점군 정합 → v4.1 깊이 일관성 검사 (`pipeline_bin_B1.yaml`) |
| 집기 결정 | 후보를 위쪽부터 시도해 처음 채택되고 파지 검사(작업영역·접근 기울기 ≤ 30°)를 통과한 것 |
| 정답 대응 | 추정 자세와 ADI가 가장 작은 하우징. TCP 오차가 ±2 mm / ±3° 안이면 올바른 채택, 밖이면 오채택 |

## 3. S2 — MuJoCo 하우징 파지 (`mujo/runtime/housing_pick.py`)

| 항목 | 구현 |
|---|---|
| 팔 | 기존 5축 팔·접촉 그리퍼 (관절 힘 ±60 Nm, 손가락 ±12 N). 손가락 간격 55 mm, 아래 연장 30 mm (관문 A dev 조정) |
| 하우징 | pactory CAD와 같은 치수의 기본 도형 합성: 몸체 원통 + 플랜지 + 다리 3개, 0.30 kg (합성 가정) |
| 자세 이식 | E = T_gt⁻¹·T_est. 집기 위치에 정답 회전으로 떨어뜨려 안정화 → T_settled. 팔은 T_settled·E만 본다 |
| 파지점 | 추정 자세가 똑바로 → 플랜지 테두리, 뒤집힘 → 몸체 중심. 손목 yaw = 다리 사이 (ψ ≡ 다리 방향 mod 60°) |
| 순서 | 접근 → 내려가기 → 집기(양쪽 패드 법선력 > 0.2 N) → 들어올림 → 이송 → 트레이에 놓기 → 안정 확인 |
| 성공 | 트레이(내부 0.16 m) 안, 바닥 높이 범위, 기울기 ≤ 15°, 속도 ≤ 5 cm/s 25틱 |
| 방향 정보 | 로봇이 믿는 방향(똑바로/뒤집힘)이 실제와 같은지 따로 기록 (`orientation_belief_correct`) |

## 4. S3 — VP 셀 (`src/factory_sim/vision_pick.py`, `configs/v5/rev_C_vp.yaml`)

| 항목 | 구현 |
|---|---|
| 위치 | 하우징 창고 옆 (`at_source`): 창고 빈 → VP는 AGV 없이, VP → KIT는 AGV |
| 1회 시도 | 촬영 0.3 s + 인식 시간(S1 표본) → 거부(빈 흔들기 5 s 후 재촬영) / 채택+파지 실패(S2 실패 시간) / 성공(S2 성공 시간) |
| 확률 | p_reject = 1 − S1 채택률, p_fail = S1 채택률 × (1 − S2 G-EST 성공률) |
| 재시도 | 최대 3회, 모두 실패하면 작업자 수동 처리 60 s (OP6·OP8, 작업자 풀·보행 포함) |
| 고장 | v2 "other" 등급 (MTBF 8 h, MTTR 20 min) |
| 난수 | (run seed, 주문 id, VP, 시도 번호) 독립 흐름 → 실패율 민감도가 같은 균등난수로 짝 비교됨 |
| 민감도 | `attempt_fail_rate` = p_reject + p_fail을 주면 두 비율을 유지해 스케일 |

## 5. 바꾼 기존 코드 (확장만)

| 파일 | 변경 | 기존 결과 영향 |
|---|---|---|
| `factory_sim/config.py` | STATION_KINDS에 vision_pick, 해당 검증 | 없음 |
| `factory_sim/engine.py` | `kind: vision_pick`이면 VP 프로세스, stats에 VP 키(있을 때만) | 없음 — V-R2 재현 검사 |
| `factory_sim/release.py` | `at_source` VP면 창고에서 바로 VP 입력으로 | 없음 — V-R2 재현 검사 |
| `mujo/envs/housing_scene.py`, `runtime/housing_pick.py` | 신규 | 기존 47개 검사 그대로 통과 |

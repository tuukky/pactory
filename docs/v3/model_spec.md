# v3 비전 가이던스 모델 명세 (1~3주차분)

작성일: 2026-10-30 / 범위: PLAN_v3 §2.1 우선순위 1~5 + 간이 합성 장면용 `cad.py`.
제외(이번 단계 아님): `synth.py`/BlenderProc 합성 데이터, 평가 세트 평가, `rev_C_vp.yaml`·`factory_sim` 연동, `dashboard_v3`.

> **합성 표기**: 하우징 CAD(`cad.py`), 간이 합성 장면, 합성 체커보드 영상, 가상 로봇 자세, 카메라·그리퍼·작업영역 값은
> 모두 **합성 가정**이다. 실제 제품 도면·설비 사양·측정값이 아니다. 실측 데이터는 공개 BOP LM-O 뿐이다.

## 1. 모듈 구성 (`src/vision_guide/`, `factory_sim`·`line_sim` import 없음)

| 파일 | 내용 |
|---|---|
| `geometry.py` | `CameraIntrinsics`, 투영·역투영(왜곡은 cv2), `depth_to_points`, SE(3) 유틸, `look_at`, `perturb_T` |
| `cad.py` | 합성 하우징 메시(`HousingSpec`), C_n 대칭, 표면 샘플, 레이캐스팅 Depth 렌더(`render_depth`), 간이 장면 |
| `segment.py` | ROI → voxel → 이상점 제거 → 법선 → 평면 RANSAC → DBSCAN → 크기 필터 |
| `pose.py` | `ObjectModel`, FPFH+RANSAC, 2단 point-to-plane ICP, 가시 비율, 대칭 후보, `run_pipeline` |
| `metrics.py` | ADD, ADD-S(ADI), re/te (BOP 정의), 대칭 고려 회전오차, 축 오차, 요약 |
| `failures.py` | 실패 사유 코드, 단계 귀속, 파지 실패 동률 규칙 |
| `datasets.py` | BOP 다운로드(zip-slip 검사), 모델·대칭, 프레임 로더, dev/eval 분할 차단 |
| `calib.py` | 합성 체커보드 렌더링, 내부 파라미터(Zhang), Hand-eye(eye-to-hand) 곡선·종단 경로 |
| `guidance.py` | T_base_tcp, 접근 기울기·작업영역 검사, 파지 선택, TCP 오차, 몬테카를로 오차 예산 |
| `run.py` | CLI (`python -m vision_guide.run <cmd>`) |

설정: `configs/v3/camera.yaml`, `pipeline.yaml`(`frozen: false`), `gripper.yaml`, `dataset.yaml`(`bop:` 절만).

## 2. 좌표계·단위·규약

- **단위**: 내부 길이 m, 각도 °(이름에 `_deg`). BOP의 mm는 `datasets.py` 경계에서 m로 바꾼다. 지표 출력은 mm·°.
- **변환 표기**: `T_a_b`는 b 좌표의 점을 a 좌표로 옮긴다 (p_a = T_a_b · p_b).
- **좌표 사슬**: `T_base_tcp = T_base_cam · T_cam_obj · T_obj_grasp`.
  - `base`: 로봇 base. 빈 바닥 = base z=0, 빈 중심 (0.6, 0, 0) m.
  - `cam`: OpenCV 축 (+x 오른쪽, +y 아래, +z 시선). 카메라는 (0.6, 0, 1.0) m에서 연직 아래를 보며 영상 위쪽 = base +x.
  - `obj`: 하우징 원점 = 몸체 축 위 바닥(다리 밑면) 중심, +z = 대칭축(플랜지 쪽).
  - `grasp`/`tcp`: TCP +z = 접근 방향(물체로 들어가는 방향).
- **픽셀 규약**: 픽셀 (u, v)는 연속 좌표 [u, u+1)×[v, v+1), 중심 (u+0.5, v+0.5).
  - OpenCV(코너 검출·`calibrateCamera`·`solvePnP`)는 픽셀 중심 = 정수 → `K_opencv()`(cx, cy −0.5)·`from_opencv()`(+0.5)로 경계 변환.
  - **BOP cam_K는 변환 없이 그대로 쓴다**: LM-O dev 프레임에서 정답 자세 레이캐스팅 실루엣과 `mask_visib` 일치율이
    그대로 0.999, +0.5 이동 시 0.95~0.99였다 (BOP 렌더러가 OpenGL 픽셀 중심 규약). `test_v3_07`이 검사한다.
- **Depth**: z-깊이(m), 결측 0. 간이 장면 렌더 광선은 `dir = K⁻¹[u+0.5, v+0.5, 1]`(z=1)이라 `t_hit`가 곧 z-깊이다.
  Depth 영상은 정합(rectified)되어 왜곡이 없다고 가정한다(`intr.pinhole()`). camera.yaml의 왜곡 계수는 RGB 체커보드 캘리브레이션 정답으로만 쓴다.
- **오차 섭동** `perturb_T(T, σ_rot, σ_trans, side="right")`: 회전벡터·병진 각 성분 ~ N(0, σ²) (축별 σ),
  `T · ΔT` — 회전은 자식 좌표 원점 주위. 자세 오차 = 물체 원점 주위, Hand-eye 오차 = 카메라 원점 주위 회전.
  축별 σ의 회전각 크기 중앙값 ≈ 1.54σ, RMS = √3σ.

## 3. 합성 하우징 CAD (`cad.py`, 합성)

`HousingSpec` 기본값(m): 몸체 r 0.035·h 0.074, 상단 플랜지 r 0.050·t 0.006, 하단 다리 3개(폭 0.014, 높이 0.012, 끝 반경 0.058).
총 높이 0.080, 직경(최대 두 점 거리) 0.1347. 원통 분할 수는 다리 수의 배수(60)라 다각형 메시도 정확히 C3 대칭이다.
부품을 이어붙인 메시(불리언 합집합 아님)이며, 모델 점 샘플은 다른 부품 안(맞닿은 면 포함) 점을 버려 외부 표면만 남긴다.

간이 장면(`make_simple_scene`): 빈 바닥 판(0.60×0.40 m) 위에 하우징 n개를 외접원 거부 샘플링으로 겹치지 않게
놓는다(똑바로 70%, 뒤집힘 30%, 무작위 yaw). 적재·가림은 없다(BlenderProc 4주차). Depth 노이즈 N(0, σ²)와 무작위 결측.

## 4. 자세 추정 파이프라인 (`pipeline.yaml`, 동결 전)

1. 역투영 → 2. ROI(깊이 0.2~2.0 m) → voxel 4 mm → statistical outlier → 법선(카메라 쪽 정렬)
3. 가장 큰 평면이 입력의 15% 이상이면 제거(거리 6 mm) → DBSCAN(eps 10 mm) → 점 수·bbox 대각(0.25~1.6 × 직경) 필터
4. 카메라에 가까운 클러스터부터 최대 10개
5. FPFH(반경 5×voxel) + RANSAC(거리 1.5×voxel, edge 0.9). **정합 방향은 관측 클러스터 → 모델**
   (fitness = 관측 점 중 모델에 맞는 비율). 재시작 5회, fitness × 가시 비율 최고 채택.
6. ICP point-to-plane 2단(15 mm → 6 mm)
7. 신뢰도: fitness ≥ 0.6 → RMSE ≤ 4 mm → 가시 비율 ≥ 0.5 (처음 실패한 검사 하나로 귀속)
   - 가시 비율: 추정 자세에서 카메라를 향한 모델 점을 z-버퍼로 투영, 관측 depth가 유효한 픽셀 중 |d − z| < 10 mm 비율.
8. 대칭 후보 {T·S}: BOP 대칭 목록(항등 포함), 하우징 C3.
9. Guidance (아래 §6).

- 후보 선택(`pick_best`): 채택(OK) 중 점수(fitness × 가시 비율) 최고, 없으면 전체 최고점 후보의 코드로 실패 귀속.
- 처리시간 키: backproject/preprocess/plane/cluster/fpfh/ransac/icp/confidence/symmetry (`perf_counter`, ms).
- **LM-O 프로파일** (`profiles.lmo`, eval-bop만 적용): 평면 거리 15 mm, 평면 최소 비율 0.10, 후보 20개.
  Kinect 테이블 depth가 6 mm 띠보다 두꺼워 평면이 둘로 갈라지는 문제 때문 (verification.md 조정 기록).
- 난수: `o3d.utility.random.seed`로 고정하지만 Open3D RANSAC은 멀티스레드라 같은 시드에서도 결과가 미세하게 달라질 수 있다.

## 5. 지표 (`metrics.py`)

- ADD, ADI(=ADD-S), re, te는 BOP toolkit `pose_error`와 같은 정의 (테스트에서 원문 전사본과 1e-12 수준 대조; toolkit 미설치라 직접 대조는 skip).
- ADD(-S): 대칭 객체(LM-O 10 eggbox, 11 glue)는 ADI, 그 외 ADD. 정답 = 오차 < 0.1 × 직경.
- 파이프라인이 실패 코드를 낸 대상은 오답(BOP의 추정 없음). `raw_rate_add_s`는 게이트를 무시한 진단용.
- 대칭 고려 회전 오차 `re_sym`, 대칭축 오차 `axis_error_deg`(축 주위 회전 제외).

## 6. 캘리브레이션·Guidance

- 체커보드: 내부 코너 9×6, 칸 25 mm. 렌더링은 픽셀당 4×4 서브샘플 광선을 왜곡 보정(캐시)해 보드 평면과 교차.
  밝기 노이즈 σ(0~255)는 가우시안. 코너 순서 180° 뒤집힘은 첫 행이 +u 방향이 되게 맞춘다(보드 자세 roll ±20° 가정).
- 내부 파라미터: `cv2.calibrateCamera`(왜곡 5계수). 투영 지도 오차 = 같은 광선을 추정·정답 모델로 투영한 픽셀 차.
- Hand-eye(eye-to-hand): 보드를 TCP에 고정(`T_tcp_board` 정답). gripper2base 자리에 inv(T_base_tcp) → 결과 = T_base_cam.
  **전처리**: base를 첫 로봇 자세 방향으로 고정 회전(W)해 풀고 되돌린다(동치 변환). TCP가 아래를 보는 자세에서
  OpenCV TSAI가 수치적으로 불안정했기 때문(verification.md).
- 노이즈 모델 (a): T_cam_board 측정에 축별 σ (오른쪽 섭동), 로봇 자세는 정확. (b) 종단: 렌더링 → 코너 → solvePnP(정답 내부 파라미터).
- 파지 목록(합성 가정): 윗면(플랜지, −z_obj 접근), 바닥면(+z_obj 접근). 접근 기울기 ≤ 30°, 작업영역 박스.
  선택 = 작업영역 → 기울기 통과 중 기울기 최소. 전부 실패 시 UNREACHABLE/APPROACH_LIMIT 다수(동률 UNREACHABLE).
- 파지 성공(합성 가정): TCP 위치 오차 ≤ 2 mm, 접근축 오차 ≤ 3° (파지 역학 아님).
- 오차 예산: 빈 중심 똑바로 놓인 하우징의 윗면 파지 기준. 자세 σ(config 0.3°/0.5 mm, 합성 가정)와
  Hand-eye σ(곡선 기준 조건 noise_level 2·15자세·PARK에서 RMS/√3)를 몬테카를로(4000)로 합성.
  허용치를 지키는 최대 Hand-eye 회전 σ는 격자 탐색, 성공률 기준 95%(gripper.yaml 사전 기재).

## 7. 데이터 (`dataset.yaml`)

- BOP LM-O (HF `bop-benchmark/lmo`: base, models, test_bop19), `data/v3/bop/lmo/` (gitignore).
- test_targets_bop19의 200장 (scene 2)을 시드 20261030으로 섞어 dev 50 (`make_split`), 나머지 미사용. 목록은 yaml에 기록.
- eval 50: dev가 아니고 같은 scene의 모든 dev 프레임과 |im_id 차| > `eval_gap_im`(5)인 후보 91장에서 같은 시드로 선택
  (`make_gapped_eval`, 연속 시퀀스 인접 프레임 누설 방지). 2026-10-30 eval 미사용 상태에서 재생성, dev 불변. dev-eval 최소 간격 6.
- 로더는 `allow_eval=True` 없이 eval을 읽지 않는다. dev 순회와 `show-bop`은 `load_frame(forbid_ids=eval)`로 eval 프레임을 막는다. CLI `eval-bop`은 `--split dev`만 받는다. `show-bop`도 eval 프레임을 거부한다.
- 주 결과 = `dbscan`(전체 파이프라인), `gt_mask`는 정답 가시 마스크 분할(정합 상한)이며 그렇게 표시한다.
- TODO(라이선스): 받은 zip에 라이선스 표기가 없다. 공개 배포 전 BOP 웹사이트 표기를 확인해 dataset.yaml에 기록.

## 8. 가정·한계 (이번 단계)

- 간이 장면은 적재·가림·조명이 없고 Depth 노이즈는 화소 독립 가우시안이다. 실제 센서 특성을 단순화했다.
- LM-O 실측 depth는 정답 렌더보다 수 mm(dev 첫 프레임 1~10 mm) 멀다(알려진 Kinect 편향). 보정하지 않았다 → 병진 오차에 그대로 포함.
- Hand-eye는 가상 로봇 자세이며 로봇 반복 정밀도를 반영하지 않는다(로봇 자세 정확 가정).
- 처리시간은 AMD Ryzen 7 3800X(8코어 16스레드), Windows 11, CPU 전용 측정값이다.

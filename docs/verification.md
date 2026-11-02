# 검증 (V01–V07) ↔ 테스트 매핑

실행: `python -m pytest tests -q`. 이 검증이 모두 통과하기 전에는 개선안 숫자를 성과로 사용하지 않는다.

| ID | 내용 | 테스트 파일 · 함수 |
|---|---|---|
| V01 | A(10)→B(15)→C(30) 결정론, 첫 양품 55초, 간격 30초, `floor((T−55)/30)+1`, T=3600에 119개 | `test_analytic_line.py`: `test_first_ship_and_interval`, `test_completed_count_formula` (T=55·84·85·3600) |
| V01 | C=25초: 첫 50초, 3600초에 143개 | `test_c25_still_bottleneck` |
| V01/V07 | 정확히 28,800초 출하 포함 → 1151개 (`run(until)` 방식은 1150) | `test_window_boundary_includes_exact_28800`, `test_window_c30` (959개) |
| — | 기준 부하 참고값 107.46 / 87.27 / 57.60 / 153.19 / 75 개/시간 | `test_nominal_capacity_baseline` |
| V02 | 각 시각 그룹 끝에서 주문 = 미투입 + WIP + 양품 + 폐기, WIP = 버퍼 + 설비 (엔진 내부 검사) | `test_conservation.py`: `test_invariants_and_status` (`check_invariants=True`) |
| V02 | 이벤트 로그 재생: Job 위치 유일, 로그 버퍼 점유 = 재생 점유 | `test_event_replay_mass_balance` |
| V02 | 종결 이벤트 Job당 1개, 버퍼 0..capacity, 미출하 분해 합 | `test_one_terminal_event_per_job`, `test_buffer_count_bounds`, `test_unshipped_decomposition` |
| V03 | 설비별 상태시간 합 = 관측시간 (0–28,800, 0–sim_end, 1e-6초) | `test_state_time_sums`, `test_early_finish_still_covers_window` |
| V03 | WIP 면적 = Job별 관측창 체류 합 | `test_wip_area_identity` |
| V03 | 시각 단조성, 설비 구간 무중첩·연속 | `test_job_time_monotonic`, `test_machine_intervals_contiguous` |
| V04 | U(5)→D(20), Q_D 용량 1: 출하 25·45·65·85·105, U 막힘 (15,25)(30,45)(50,65) 합 40, 막힘 중 새 작업 없음 | `test_blocking.py`: `test_blocking_capacity_1` |
| V04 | 용량 2: 출하 동일, 막힘 (20,25)(30,45) 합 20 | `test_blocking_capacity_2` |
| V04 | 막힌 Job은 하류 버퍼 점유에 포함되지 않음 | `test_blocked_job_not_counted_in_buffer` |
| V05 | 불합격 0 → 재작업·폐기 0 | `test_quality.py`: `test_no_defects` |
| V05 | 불합격 1·회복 1 → 전량 1회 재작업 후 출하 | `test_all_fail_all_recover` |
| V05 | 불합격 1·회복 0 → 전량 폐기, 출하 0, PACK 진입 0 | `test_all_fail_none_recover` |
| V05 | 재작업 ≤ 1회, Job ID 유지, `fail_u < fail_rate`인 Job만 불합격 | `test_rework_at_most_once_and_ids_kept`, `test_fail_draw_decides_inspection` |
| V06 | 2대 독립 준비 기억: job1→ASM_1(30초), job2→ASM_2 준비 0, job3→ASM_1 SETUP 136–176, PROCESSING 176–286 | `test_parallel_setup.py`: `test_independent_setup_memory` |
| V06 | 1대 [P1,P1,P1] 준비 0회, [P1,P3] 준비 40초 | `test_single_machine_setup` |
| V06 | 2대 간 겹침 존재, 동일 설비 안 중첩 없음 | `test_parallel_overlap_allowed_but_not_within_machine` |
| V07 | 같은 설정·시드 → events/jobs/intervals/metrics 동일 (run_id 제외), 저장 디렉토리 재계산 일치 | `test_reproducibility.py`: `test_same_seed_identical`, `test_metrics_from_saved_dir_match` |
| V07 | PYTHONHASHSEED가 달라도 난수·순열 동일 | `test_draws_independent_of_hash_seed` |
| V07 | B0 vs S5: Job별 기본 시간·품질 난수 동일, S5 품종 연속 투입 | `test_common_random_numbers_b0_vs_s5` |
| V07 | S3: ASM 가공시간 = B0 기본 표본 × 0.8, 준비시간 동일 | `test_time_scale_applies_to_processing_only` |
| V07 | 첫 투입 0초, 투입 간격 ≥ 48초, WIP ≤ 40 | `test_release_interval_and_wip_limit` |
| V07 | 제한시각 20,000초 → `incomplete_at_limit`, 종결 KPI null, 예외 없음, 보존식 성립 | `test_time_limit_incomplete` |
| 설정·통계 | 설정 검증 차단, 데이터 CSV = baseline, mean_ci 손계산, paired 짝, 개선율, 조합 규칙 | `test_config_and_stats.py` |

## 최근 실행 결과

| 일시 | 커밋 | 결과 |
|---|---|---|
| TBD | TBD | TBD |

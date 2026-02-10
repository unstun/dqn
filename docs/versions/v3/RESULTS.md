# v3 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.5` / `0.1`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `32.86` / `576.928`
- CNN 路径时间(s) short/long： `12.02` / `55.3`
- CNN argmax 不可行动作率 short/long： `0.275` / `0.504`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.5`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403/table2_kpis_mean_raw.csv`
- 2. SR(short/long)=`0.5`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0/20260209_123212` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0/20260209_123212/table2_kpis_mean_raw.csv`
- 3. SR(short/long)=`0.3`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast5pre_h20mp0_ms1200/20260209_124454` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast5pre_h20mp0_ms1200/20260209_124454/table2_kpis_mean_raw.csv`
- 4. SR(short/long)=`0.2`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast6pre_h20mp0/20260209_125209` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast6pre_h20mp0/20260209_125209/table2_kpis_mean_raw.csv`
- 5. SR(short/long)=`0.2`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast6pre_h5mp0/20260209_130104` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast6pre_h5mp0/20260209_130104/table2_kpis_mean_raw.csv`
- 6. SR(short/long)=`0.1`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h15mp0/20260209_121341` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h15mp0/20260209_121341/table2_kpis_mean_raw.csv`
- 7. SR(short/long)=`0.1`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h20mp0/20260209_121051` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h20mp0/20260209_121051/table2_kpis_mean_raw.csv`
- 8. SR(short/long)=`0.1`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h10mp0/20260209_121222` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast3_h10mp0/20260209_121222/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

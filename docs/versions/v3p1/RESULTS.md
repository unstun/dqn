# v3p1 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827/infer/20260209_145935/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.7` / `0.2`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `42.905` / `283.09`
- CNN 路径时间(s) short/long： `20.721` / `35.475`
- CNN argmax 不可行动作率 short/long： `0.19` / `0.434`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.7`/`0.2` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_full_finalckpt/train_20260209_143827/infer/20260209_145935/table2_kpis_mean_raw.csv`
- 2. SR(short/long)=`0.3`/`0.5` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_finalckpt/train_20260209_142238` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_finalckpt/train_20260209_142238/infer/20260209_143207/table2_kpis_mean_raw.csv`
- 3. SR(short/long)=`0.7`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke/train_20260209_130822` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke/train_20260209_130822/infer/20260209_131906/table2_kpis_mean_raw.csv`
- 4. SR(short/long)=`0.4`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_infer_gridpow2/20260209_141649` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_infer_gridpow2/20260209_141649/table2_kpis_mean_raw.csv`
- 5. SR(short/long)=`0.2`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_finalckpt_h10/20260209_143422` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_finalckpt_h10/20260209_143422/table2_kpis_mean_raw.csv`
- 6. SR(short/long)=`0.2`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_auto/train_20260209_150200` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p1_smoke_auto/train_20260209_150200/infer/20260209_151903/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

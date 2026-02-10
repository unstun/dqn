# v2 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246/infer/20260209_083902/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0` / `0`
- Hybrid 成功率 short/long： `1` / `1`
- CNN 规划代价 short/long： `inf` / `inf`
- CNN 路径时间(s) short/long： `N/A` / `N/A`
- CNN argmax 不可行动作率 short/long： `0.465` / `0.443`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246/infer/20260209_083902/table2_kpis_mean_raw.csv`
- 2. SR(short/long)=`0`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_084511` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_084511/infer/20260209_085139/table2_kpis_mean_raw.csv`
- 3. SR(short/long)=`0`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke_h3_margin/train_20260209_090206` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke_h3_margin/train_20260209_090206/infer/20260209_090759/table2_kpis_mean_raw.csv`
- 4. SR(short/long)=`0`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke_finalckpt/train_20260209_085424` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke_finalckpt/train_20260209_085424/infer/20260209_090032/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

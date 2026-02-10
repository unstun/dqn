# v3p5 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207/infer/20260209_162247/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.2` / `0`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `104.373` / `inf`
- CNN 路径时间(s) short/long： `12.15` / `N/A`
- CNN argmax 不可行动作率 short/long： `0.553` / `0.49`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.2`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p5_smoke/train_20260209_160207/infer/20260209_162247/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

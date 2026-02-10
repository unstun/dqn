# v3p2 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603/infer/20260209_135155/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.2` / `0.1`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `76.071` / `536.724`
- CNN 路径时间(s) short/long： `9.875` / `44.4`
- CNN argmax 不可行动作率 short/long： `0.156` / `0.141`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.2`/`0.1` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p2_smoke_ep1/train_20260209_133603/infer/20260209_135155/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

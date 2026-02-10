# v3p3 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600/infer/20260209_141406/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.2` / `0.2`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `126.289` / `251.25`
- CNN 路径时间(s) short/long： `28` / `38.25`
- CNN argmax 不可行动作率 short/long： `0.326` / `0.269`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.2`/`0.2` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p3_smoke/train_20260209_135600/infer/20260209_141406/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

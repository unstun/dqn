# v3p4 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029/infer/20260209_155954/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.1` / `0`
- Hybrid 成功率 short/long： `0.9` / `1`
- CNN 规划代价 short/long： `113.285` / `inf`
- CNN 路径时间(s) short/long： `6.9` / `N/A`
- CNN argmax 不可行动作率 short/long： `0.312` / `0.422`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0.1`/`0` | run=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029` | kpi=`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p4_smoke/train_20260209_154029/infer/20260209_155954/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

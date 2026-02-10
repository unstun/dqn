# v1 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017/infer/20260209_003447/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0` / `0`
- Hybrid 成功率 short/long： `1` / `1`
- CNN 规划代价 short/long： `inf` / `inf`
- CNN 路径时间(s) short/long： `N/A` / `N/A`
- CNN argmax 不可行动作率 short/long： `0.356` / `0.194`

## 本版本观察到的代表运行（正式集合）
- 1. SR(short/long)=`0`/`0` | run=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | kpi=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017/infer/20260209_003447/table2_kpis_mean_raw.csv`
- 2. SR(short/long)=`0`/`0` | run=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_230823` | kpi=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_230823/infer/20260208_231532/table2_kpis_mean_raw.csv`
- 3. SR(short/long)=`0`/`0` | run=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_234844` | kpi=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_234844/infer/20260208_235956/table2_kpis_mean_raw.csv`
- 4. SR(short/long)=`0`/`0` | run=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_231843` | kpi=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260208_231843/infer/20260208_232836/table2_kpis_mean_raw.csv`
- 5. SR(short/long)=`0`/`0` | run=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_000722` | kpi=`runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_000722/infer/20260209_001836/table2_kpis_mean_raw.csv`

## 结论
- 对照 Hybrid A*-MPC 最终门槛未通过。

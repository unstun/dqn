# v3p11 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/infer/20260209_202232/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.3` / `0.5`
- Hybrid 成功率 short/long： `0.9` / `1.0`
- CNN 规划代价 short/long： `65.701` / `98.545`
- CNN 路径时间(s) short/long： `11.65` / `30.68`
- CNN argmax 不可行动作率 short/long： `0.239` / `0.156`

## 门槛检查
- `success_rate(CNN) >= success_rate(Hybrid)`: short `FAIL`, long `FAIL`
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `FAIL`, long `FAIL`
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `FAIL`, long `FAIL`
- 最终门槛状态：`未通过`

## 失败分析（来自 table2_kpis_raw.csv）
- short: `timeout=6`, `reached=3`, `collision=1`
- long: `reached=5`, `timeout=4`, `collision=1`
- 主要瓶颈：short 套件 timeout 仍占主导。

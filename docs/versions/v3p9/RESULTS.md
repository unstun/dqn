# v3p9 - 结果

## 代表性正式指标
- KPI 路径： `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p9_smoke/train_20260209_184524/infer/20260209_190223/table2_kpis_mean_raw.csv`
- CNN 成功率 short/long： `0.2` / `0.2`
- Hybrid 成功率 short/long： `0.9` / `1.0`
- CNN 规划代价 short/long： `104.655` / `237.684`
- CNN 路径时间(s) short/long： `16.85` / `26.225`
- CNN argmax 不可行动作率 short/long： `0.305` / `0.264`

## 门槛检查
- `success_rate(CNN) >= success_rate(Hybrid)`: short `FAIL`, long `FAIL`
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `FAIL`, long `FAIL`
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `FAIL`, long `FAIL`
- 最终门槛状态：`未通过`

## 失败分析（来自 table2_kpis_raw.csv）
- short: `timeout=7`, `reached=2`, `collision=1`
- long: `timeout=7`, `reached=2`, `collision=1`
- 主要瓶颈：超时崩塌 仍占主导。

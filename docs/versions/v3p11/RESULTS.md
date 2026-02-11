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

## Re-eval（strict-argmax, fixed pairs20 v1，2026-02-10）

固定 pairs（公平对比；short/long 各 20）：
- short: `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- long: `configs/repro_20260210_forest_a_pairs_long20_v1.json`

### short（strict-argmax）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_strict_short_pairs20_v1 --forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_strict_short_pairs20_v1/20260210_222342`
- run_json: `runs/repro_20260210_reval_v3p11_strict_short_pairs20_v1/20260210_222342/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_strict_short_pairs20_v1/20260210_222342/table2_kpis_mean_raw.csv`
- 结果：SR=`0.05`；`argmax_inadmissible_rate=0.427`
- failure_reason（raw，CNN-DDQN）：`collision=19, reached=1`

### long（strict-argmax）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::long --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_strict_long_pairs20_v1 --forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_strict_long_pairs20_v1/20260210_222407`
- run_json: `runs/repro_20260210_reval_v3p11_strict_long_pairs20_v1/20260210_222407/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_strict_long_pairs20_v1/20260210_222407/table2_kpis_mean_raw.csv`
- 结果：SR=`0.05`；`argmax_inadmissible_rate=0.217`
- failure_reason（raw，CNN-DDQN）：`collision=19, reached=1`

### 汇总（strict-argmax, fixed pairs20 v1）
- SR short/long：`0.05 / 0.05`

## Re-eval（hybrid/shielded, fixed pairs20 v1，2026-02-10）

说明：本口径使用 `--no-forest-no-fallback`，允许推理期 replacement/override/fallback；不得与 `strict-argmax` 混称。

### short（hybrid/shielded）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_hybrid_short_pairs20_v1 --no-forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_v1/20260210_222432`
- run_json: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_v1/20260210_222432/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_v1/20260210_222432/table2_kpis_mean_raw.csv`
- 结果：SR=`0.75`；`argmax_inadmissible_rate=0.200`
- failure_reason（raw，CNN-DDQN）：`reached=15, collision=3, timeout=2`

### long（hybrid/shielded）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::long --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_hybrid_long_pairs20_v1 --no-forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_v1/20260210_222513`
- run_json: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_v1/20260210_222513/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_v1/20260210_222513/table2_kpis_mean_raw.csv`
- 结果：SR=`0.85`；`argmax_inadmissible_rate=0.210`
- failure_reason（raw，CNN-DDQN）：`reached=17, collision=2, timeout=1`

### 汇总（hybrid/shielded, fixed pairs20 v1）
- SR short/long：`0.75 / 0.85`

## Re-eval（hybrid/shielded, no-heuristic-fallback, fixed pairs20 v1，2026-02-10）

说明：本轮保持 `--no-forest-no-fallback`，但代码已移除“启发式 fallback rollout takeover”。
推理期保留 `stop-override + top-k/mask replacement`；当 top-k 与 mask 都无解时，不再触发 heuristic fallback。

### short（hybrid/shielded, no-heur）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_hybrid_short_pairs20_noheur_v1 --no-forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_noheur_v1/20260210_231222`
- run_json: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_noheur_v1/20260210_231222/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_hybrid_short_pairs20_noheur_v1/20260210_231222/table2_kpis_mean_raw.csv`
- 结果：SR=`0.75`；`argmax_inadmissible_rate=0.200`；`fallback_rate=0.000`
- failure_reason（raw，CNN-DDQN）：`reached=15, collision=3, timeout=2`

### long（hybrid/shielded, no-heur）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --baselines --envs forest_a::long --no-rand-two-suites --random-start-goal --runs 20 --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json --models runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models --out repro_20260210_reval_v3p11_hybrid_long_pairs20_noheur_v1 --no-forest-no-fallback`
- run_dir: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_noheur_v1/20260210_231255`
- run_json: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_noheur_v1/20260210_231255/configs/run.json`
- kpi: `runs/repro_20260210_reval_v3p11_hybrid_long_pairs20_noheur_v1/20260210_231255/table2_kpis_mean_raw.csv`
- 结果：SR=`0.85`；`argmax_inadmissible_rate=0.210`；`fallback_rate=0.000`
- failure_reason（raw，CNN-DDQN）：`reached=17, collision=2, timeout=1`

### 汇总（hybrid/shielded, no-heuristic-fallback, fixed pairs20 v1）
- SR short/long：`0.75 / 0.85`

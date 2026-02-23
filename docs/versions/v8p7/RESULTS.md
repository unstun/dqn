# v8p7 结果对比（infer-only smoke 已回填；train+infer smoke 待跑）

> 说明：本版是推理侧补丁（接近目标速度整形）。回填结果时必须写清 `forest_replace_topq` 与 `forest_goal_approach_*` 的取值，并与 `Hybrid A*-MPC`（基线）同对比。

## 1) 关键工件路径

- infer-only smoke（固定 `v8p6` checkpoint；runs=3；CNN-DDQN + Hybrid baseline）：
  - profile：`configs/repro_20260223_v8p7_goal_approach_infer_smoke.json`
  - models：`runs/v8p6_replace_topq_smoke/train_20260223_191450/models`
  - run_dir：`runs/v8p7_goal_approach_infer_smoke/20260223_230524`
  - mean KPI：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/table2_kpis_mean_raw.csv`
  - raw KPI：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/table2_kpis_raw.csv`
- train+infer smoke（episodes=150, runs=3）：`N/A`

## 2) short/mid/long KPI（infer-only smoke；runs=3，mean）

读取：`runs/v8p7_goal_approach_infer_smoke/20260223_230524/table2_kpis_mean_raw.csv`。

| suite | algo | success_rate | avg_path_length | path_time_s | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 15.4331 | 9.7667 | 0.095 | 0.095 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | N/A | N/A |
| mid | CNN-DDQN | 1.000 | 24.3699 | 16.0167 | 0.179 | 0.179 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | N/A | N/A |
| long | CNN-DDQN | 1.000 | 43.5512 | 25.7000 | 0.131 | 0.131 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | N/A | N/A |

小结：
- **硬约束（SR≈1.0）已满足**：short/mid/long 均 `SR=1.0`。
- 路径长度：short 优于 baseline；mid/long 与 baseline 接近。
- 路径时间：mid/long 仍落后 baseline（本版下一步优先优化点）。

## 3) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`


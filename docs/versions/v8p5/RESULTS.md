# v8p5 结果对比（待回填）

> 说明：本版是 replace-ranking 消融开关引入，默认行为不变；因此结果需要明确写清 `forest_replace_ranking` 取值与评测样本集。

## 1) 关键工件路径

- 回归（fixed pairs，replace-ranking 消融）：
  - profile：`configs/repro_20260223_v8p5_replace_ranking_regression.json`
  - pairs：`configs/pairs_v8p3_smoke_failures.json`
  - `progress_clearance_q`：`runs/v8p5_replace_ranking_regression/20260222_222704`
  - `clearance_progress_q`：`runs/v8p5_replace_ranking_regression/20260222_223308`
  - `q`（基线）：`runs/v8p5_replace_ranking_regression/20260222_223339`
- smoke（episodes=150, runs=3）：`N/A`
  - profile：`configs/v8p5.json`

## 2) fixed-pairs KPI（回归）

读取：各 run_dir 下的 `table2_kpis_mean_raw.csv` / `table2_kpis_raw.csv`。

| ranking | run_dir | mid SR | mid avg_path_length | mid path_time_s | long SR | long avg_path_length | long path_time_s | failures |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `progress_clearance_q` | `runs/v8p5_replace_ranking_regression/20260222_222704` | 1.000 | 35.4491 | 20.7000 | 1.000 | 35.4491 | 20.7000 | reached=4/4 |
| `clearance_progress_q` | `runs/v8p5_replace_ranking_regression/20260222_223308` | 1.000 | 37.7343 | 23.1000 | 1.000 | 37.7343 | 23.1000 | reached=4/4 |
| `q` | `runs/v8p5_replace_ranking_regression/20260222_223339` | 0.000 | N/A | N/A | 0.000 | N/A | N/A | collision=2/4, timeout=2/4 |

结论：在该 fixed-pairs 上，`q` 仍复现 `collision/timeout`；两种 tie-break 策略均能修复失败，其中 `progress_clearance_q` 的 path/time 更小。

### baseline 对比（同一 fixed-pairs，`progress_clearance_q` vs `Hybrid A*-MPC`）

读取：`runs/v8p5_replace_ranking_regression/20260222_224400/table2_kpis_mean_raw.csv`

| env | algo | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| mid | CNN-DDQN（`progress_clearance_q`） | 1.000 | 35.4491 | 20.7000 | 0.113748 |
| mid | Hybrid A*-MPC | 1.000 | 34.4898 | 18.5500 | 0.057540 |
| long | CNN-DDQN（`progress_clearance_q`） | 1.000 | 35.4491 | 20.7000 | 0.113748 |
| long | Hybrid A*-MPC | 1.000 | 34.4898 | 18.5500 | 0.057540 |

当前结论：在该 fixed-pairs 上，baseline 仍略优（更短/更快/更平滑）；但 RL 在保持 SR=1.0 的前提下已接近 baseline，且 `inference_time_s` 显著更低（RL < baseline）。

## 3) short/mid/long KPI（smoke）

- `table2_kpis_mean_raw.csv`：`N/A`
- `failure_reason` 分布（来自 `table2_kpis_raw.csv`）：`N/A`

## 4) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`

## 5) 结论（待回填）

- fixed-pairs 回归已通过（mid collision + long timeout 对应样本均 reach）。
- 仍需 smoke 验证其对随机起终点分布的泛化收益与稳定性。

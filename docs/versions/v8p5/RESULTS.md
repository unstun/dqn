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

## 3.1) short/mid/long KPI（infer-only smoke，固定 `v7p1` checkpoint）

- profile：`configs/repro_20260223_v8p5_replace_ranking_infer_smoke.json`
- models：`runs/v7p1_train300_esbest/train_20260221_010743`
- runs（replace-ranking 消融）：
  - `q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217`
  - `progress_clearance_q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252`
  - `clearance_progress_q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327`

### CNN-DDQN（runs=3，mean）

| ranking | run_dir | short SR | short L | short T | mid SR | mid L | mid T | long SR | long L | long T | long inad | long fb | failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `q` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172217` | 1.000 | 15.9569 | 9.9667 | 1.000 | 25.0974 | 15.9500 | 1.000 | 52.5492 | 31.7333 | 0.257 | 0.257 | reached=9/9 |
| `progress_clearance_q` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172252` | 0.667 | 14.7217 | 9.7500 | 1.000 | 24.7806 | 15.3500 | 1.000 | 46.6938 | 26.5000 | 0.143 | 0.143 | short collision=1/3 |
| `clearance_progress_q` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172327` | 0.667 | 14.8694 | 9.9250 | 1.000 | 25.0159 | 16.7833 | 1.000 | 49.1143 | 29.4333 | 0.183 | 0.183 | short collision=1/3 |

注：
- `L` = `avg_path_length`（越小越好）
- `T` = `path_time_s`（越小越好）
- `inad` = `argmax_inadmissible_rate`
- `fb` = `fallback_rate`

### Hybrid A*-MPC（同一随机对；runs=3，mean）

读取：任一 run_dir 下 `table2_kpis_mean_raw.csv` 的 Hybrid 行（本轮三次 baseline 一致）。

| suite | success_rate | avg_path_length | path_time_s |
|---|---:|---:|---:|
| short | 1.000 | 17.0342 | 10.2667 |
| mid | 1.000 | 24.0814 | 13.3333 |
| long | 1.000 | 43.0107 | 22.8167 |

当前结论（infer-only）：`progress_clearance_q` / `clearance_progress_q` 能明显压 long 的 L/T（相对 `q`），但 short 出现碰撞回潮（`collision=1/3`），不满足 `SR≈1.0` 的硬约束；因此暂不建议在随机分布上启用 tie-break 作为默认策略。

## 4) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`

## 5) 结论（待回填）

- fixed-pairs 回归已通过（mid collision + long timeout 对应样本均 reach）。
- 仍需 smoke 验证其对随机起终点分布的泛化收益与稳定性。

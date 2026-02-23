# v8p6 结果对比（infer-only smoke 已回填；train+infer smoke 待跑）

> 说明：本版新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束），默认不启用（`0`）。回填结果时必须写清 `forest_replace_ranking` 与 `forest_replace_topq` 的取值。

## 1) 关键工件路径

- infer-only smoke（固定 `v7p1` checkpoint；同一随机对；runs=3；`forest_replace_ranking=progress_clearance_q`）：
  - profile：`configs/repro_20260223_v8p6_replace_topq_infer_smoke.json`
  - models：`runs/v7p1_train300_esbest/train_20260221_010743`
  - topq=2（默认）：`runs/v8p6_replace_topq_infer_smoke/20260223_185519`
  - topq=1（≈纯 Q replacement 对照）：`runs/v8p6_replace_topq_infer_smoke/20260223_185553`
  - topq=3（本轮更优候选）：`runs/v8p6_replace_topq_infer_smoke/20260223_185628`
- train+infer smoke（episodes=150, runs=3）：
  - profile：`configs/v8p6.json`
  - train_run：`N/A`
  - infer_run：`N/A`

## 2) short/mid/long KPI（infer-only smoke；CNN-DDQN runs=3，mean）

读取：各 run_dir 下 `table2_kpis_mean_raw.csv` / `table2_kpis_raw.csv`。

### CNN-DDQN（runs=3，mean）

| replace_topq | run_dir | short SR | short L | short T | mid SR | mid L | mid T | long SR | long L | long T | long inad | long fb | failures |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | `runs/v8p6_replace_topq_infer_smoke/20260223_185519` | 1.000 | 16.2665 | 10.5167 | 1.000 | 24.4362 | 15.0667 | 1.000 | 47.0414 | 27.0167 | 0.171 | 0.171 | reached=9/9 |
| 1 | `runs/v8p6_replace_topq_infer_smoke/20260223_185553` | 1.000 | 15.9569 | 9.9667 | 1.000 | 25.0974 | 15.9500 | 1.000 | 52.5492 | 31.7333 | 0.257 | 0.257 | reached=9/9 |
| 3 | `runs/v8p6_replace_topq_infer_smoke/20260223_185628` | 1.000 | 15.8815 | 10.1333 | 1.000 | 25.7448 | 15.2333 | 1.000 | 45.7103 | 25.3333 | 0.141 | 0.141 | reached=9/9 |

注：
- `L` = `avg_path_length`（越小越好）
- `T` = `path_time_s`（越小越好）
- `inad` = `argmax_inadmissible_rate`
- `fb` = `fallback_rate`

### Hybrid A*-MPC baseline（同一随机对；runs=3，mean）

读取：任一 run_dir 下 `table2_kpis_mean_raw.csv` 的 Hybrid 行（三次相同）。

| suite | success_rate | avg_path_length | path_time_s |
|---|---:|---:|---:|
| short | 1.000 | 17.0342 | 10.2667 |
| mid | 1.000 | 24.0814 | 13.3333 |
| long | 1.000 | 43.0107 | 22.8167 |

### ablation 小结（vs topq=1）

- topq=2：long `L/T` 从 `52.5492/31.7333` → `47.0414/27.0167`（Δ`-5.5078/-4.7166`）；三套件均值 `L/T` Δ`-1.9531/-1.6833`。
- topq=3：long `L/T` 从 `52.5492/31.7333` → `45.7103/25.3333`（Δ`-6.8389/-6.4000`）；三套件均值 `L/T` Δ`-2.0890/-2.3167`。
- 本轮（infer-only）更优候选：topq=3（`SR=1.0` 前提下，三套件均值 `L/T` 最小；且 long 的 `inad/fb` 更低）。

## 3) short/mid/long KPI（train+infer smoke）

- `table2_kpis_mean_raw.csv`：`N/A`
- `failure_reason` 分布：`N/A`

## 4) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`

## 5) 结论（当前仅含 infer-only smoke）

- infer-only smoke：topq=2/3 在该随机样本上均通过 smoke 门（short/mid/long `SR=1.0`），且相对 topq=1（≈纯 Q replacement）明显压低 long `avg_path_length/path_time_s`。
- 当前仍未超过 baseline（Hybrid A*-MPC）的 mid/long `avg_path_length/path_time_s`；下一步应继续跑 `v8p6` 的 train+infer smoke（episodes=150, runs=3），验证训练期同开关是否带来稳定收益。

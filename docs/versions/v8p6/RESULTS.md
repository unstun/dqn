# v8p6 结果对比（infer-only smoke 已回填；train+infer smoke 已跑：NO-GO）

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
  - train_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450`
  - infer_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545`

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

- `table2_kpis_mean_raw.csv`：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545/table2_kpis_mean_raw.csv`
- `failure_reason` 分布（来自 `table2_kpis_raw.csv`）：
  - short：collision=1/3
  - mid：collision=1/3
  - long：reached=3/3

运行说明：
- 命令：`conda run -n ros2py310 python train.py --profile v8p6 --forest-replace-topq 3` + `conda run -n ros2py310 python infer.py --profile v8p6 --forest-replace-topq 3`
- 训练 run_dir：`runs/v8p6_replace_topq_smoke/train_20260223_191450`（episodes=140/150 early-stop）
- 推理 run_dir：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545`

### CNN-DDQN vs Hybrid A*-MPC（runs=3，mean）

| suite | algo | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 15.2801 | 9.6250 | 0.110721 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 24.4429 | 14.2750 | 0.066587 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 1.000 | 44.3106 | 25.6000 | 0.111461 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 4) 门槛检查（最终门槛仅供格式，未评测）

- short（runs=20）：`N/A`
- long（runs=20）：`N/A`

## 5) 结论（infer-only + train+infer smoke）

- infer-only smoke：topq=2/3 在该随机样本上均通过 smoke 门（short/mid/long `SR=1.0`），且相对 topq=1（≈纯 Q replacement）明显压低 long `avg_path_length/path_time_s`。
- train+infer smoke（episodes=150）：在该次训练产物上 **short/mid 均出现 collision=1/3（SR=0.667）**，未通过 smoke 门（NO-GO）；同时 mid/long 的 `avg_path_length/path_time_s` 仍落后 baseline。

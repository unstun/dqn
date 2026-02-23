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

## 6) 固定碰撞对回放 + 消融（诊断；runs=1）

目的：定位 v8p6 训练产物在 short/mid 的 `collision` 触发态，并在固定 (start,goal) 条件下，对推理侧开关做最小消融，寻找能“消除碰撞”的候选参数组合。

### 6.1 配置与工件

- profiles：
  - `configs/repro_20260223_v8p6_short_collision_ablation.json`
  - `configs/repro_20260223_v8p6_mid_collision_ablation.json`
- fixed pairs：
  - `configs/pairs_v8p6_train_smoke_short_collision.json`
  - `configs/pairs_v8p6_train_smoke_mid_collision.json`
- models：`runs/v8p6_replace_topq_smoke/train_20260223_191450/models`
- 回归 run（不改动参数；`replace_topq=3, min_od=0.02, turn_penalty=0.0`；已开启 traces）：
  - short：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_211314`
  - mid：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_211419`

### 6.2 关键现象（来自 traces）

两次碰撞都发生在 **进入 goal 区域附近的最后一步**（而非早期撞树）：

- short 回归：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_211314/traces/forest_a_short__CNN-DDQN__run0.csv`
  - `collision_first_step=198`，且 `d_goal_m≈0.85 < goal_tolerance_m=1.0`（已在 goal pose 区附近，但尚未满足 stop/straighten 条件）
- mid 回归：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_211419/traces/forest_a_mid__CNN-DDQN__run0.csv`
  - `collision_first_step=378`，且 `d_goal_m≈0.50 < 1.0`

推断：碰撞更像是“靠近 goal 后的停稳/摆正阶段”触发（与 `stop override`（到达 goal pose 后强制执行停止动作）有关），而不是早期的安全屏蔽完全失效。

### 6.3 推理侧消融网格（每点 runs=1）

说明：
- 这是诊断回路：`runs=1` 仅用于定位现象，不代表可泛化结论。
- 对于 `collision/timeout`，表中的 `L/T` 是 **失败前的部分轨迹** 指标，仅用于对比，不可直接当作性能收益。

#### short（固定 pair：`configs/pairs_v8p6_train_smoke_short_collision.json`）

| topq | min_od | turn_pen | CNN reason | CNN L | CNN T | CNN inad | CNN fb | base reason | base L | base T |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.02 | 0.00 | collision | 18.1969 | 9.9 | 0.086 | 0.086 | collision | 18.7852 | 12.75 |
| 0 | 0.02 | 0.20 | collision | 18.1969 | 9.9 | 0.086 | 0.086 | collision | 18.7852 | 12.75 |
| 0 | 0.05 | 0.00 | collision | 18.1969 | 9.9 | 0.086 | 0.086 | collision | 18.7852 | 12.75 |
| 0 | 0.05 | 0.20 | collision | 18.1969 | 9.9 | 0.086 | 0.086 | collision | 18.7852 | 12.75 |
| 1 | 0.02 | 0.00 | collision | 17.8449 | 9.8 | 0.138 | 0.138 | collision | 18.7852 | 12.75 |
| 1 | 0.02 | 0.20 | reached | 18.7948 | 11.6 | 0.142 | 0.142 | collision | 18.7852 | 12.75 |
| 1 | 0.05 | 0.00 | collision | 17.9373 | 9.85 | 0.147 | 0.147 | collision | 18.7852 | 12.75 |
| 1 | 0.05 | 0.20 | reached | 18.7948 | 11.6 | 0.142 | 0.142 | collision | 18.7852 | 12.75 |
| 3 | 0.02 | 0.00 | collision | 18.1257 | 9.9 | 0.106 | 0.106 | collision | 18.7852 | 12.75 |
| 3 | 0.02 | 0.20 | timeout | 20.7148 | 60 | 0.149 | 0.149 | collision | 18.7852 | 12.75 |
| 3 | 0.05 | 0.00 | collision | 18.4332 | 10.05 | 0.095 | 0.095 | collision | 18.7852 | 12.75 |
| 3 | 0.05 | 0.20 | timeout | 19.4731 | 60 | 0.163 | 0.163 | collision | 18.7852 | 12.75 |

注：本轮诊断中 baseline（Hybrid A*-MPC）在该 short 固定 pair 上多次出现 `collision`，与之前 smoke 结果（baseline reached）不一致，后续需单独排查 baseline 的可复现性/确定性（例如 planner/controller 的 tie-break 或数值差异）。

#### mid（固定 pair：`configs/pairs_v8p6_train_smoke_mid_collision.json`）

| topq | min_od | turn_pen | CNN reason | CNN L | CNN T | CNN inad | CNN fb | base reason | base L | base T |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.02 | 0.00 | reached | 25.502 | 18.25 | 0.356 | 0.356 | reached | 24.2317 | 14.8 |
| 0 | 0.02 | 0.20 | reached | 25.502 | 18.25 | 0.356 | 0.356 | reached | 24.2317 | 14.8 |
| 0 | 0.05 | 0.00 | reached | 25.5189 | 17.1 | 0.322 | 0.322 | reached | 24.2317 | 14.8 |
| 0 | 0.05 | 0.20 | reached | 25.5189 | 17.1 | 0.322 | 0.322 | reached | 24.2317 | 14.8 |
| 1 | 0.02 | 0.00 | timeout | 25.5647 | 60 | 0.143 | 0.143 | reached | 24.2317 | 14.8 |
| 1 | 0.02 | 0.20 | timeout | 28.107 | 60 | 0.789 | 0.789 | reached | 24.2317 | 14.8 |
| 1 | 0.05 | 0.00 | reached | 24.0555 | 16.6 | 0.398 | 0.398 | reached | 24.2317 | 14.8 |
| 1 | 0.05 | 0.20 | reached | 27.7181 | 21.1 | 0.422 | 0.422 | reached | 24.2317 | 14.8 |
| 3 | 0.02 | 0.00 | collision | 26.3424 | 18.9 | 0.291 | 0.291 | reached | 24.2317 | 14.8 |
| 3 | 0.02 | 0.20 | reached | 26.2504 | 19.7 | 0.322 | 0.322 | reached | 24.2317 | 14.8 |
| 3 | 0.05 | 0.00 | reached | 24.5839 | 16.45 | 0.301 | 0.301 | reached | 24.2317 | 14.8 |
| 3 | 0.05 | 0.20 | timeout | 3.0907 | 60 | 0.953 | 0.953 | reached | 24.2317 | 14.8 |

### 6.4 当前“同时通过 short+mid 固定碰撞对”的交集候选

- `forest_replace_topq=1` + `forest_min_od_m=0.05` + `forest_topk_turn_penalty=0.2`：
  - short：`reached`（`runs/v8p6_ablate_short_topq1_od0p05_tp0p2`）
  - mid：`reached`（`runs/v8p6_ablate_mid_topq1_od0p05_tp0p2`）

该组合目前仅用于“消除碰撞”的诊断候选；是否满足 “SR≈1.0 前提下压 L/T” 仍需回到随机分布 smoke/full 门上验证。

### 6.5 随机分布 smoke 验证（runs=3；回到随机起终点）

目的：验证 6.4 的“诊断候选”在随机分布（short/mid/long）下是否仍能维持 `SR≈1.0`，并观察 `avg_path_length/path_time_s` 相对 baseline 的变化。

- models：`runs/v8p6_replace_topq_smoke/train_20260223_191450/models`（v8p6 train+infer smoke 训练产物）
- 命令：
  - `conda run -n ros2py310 python infer.py --profile v8p6 --models runs/v8p6_replace_topq_smoke/train_20260223_191450/models --out v8p6_candidate_infer_smoke_topq1_od0p05_tp0p2 --forest-replace-topq 1 --forest-min-od-m 0.05 --forest-topk-turn-penalty 0.2`
- run_dir：`runs/v8p6_candidate_infer_smoke_topq1_od0p05_tp0p2/20260223_215001`

读取：`table2_kpis_mean_raw.csv` / `table2_kpis_raw.csv`。

| suite | algo | success_rate | avg_path_length | path_time_s | failures |
|---|---|---:|---:|---:|---|
| short | CNN-DDQN | 1.000 | 17.3668 | 14.9833 | reached=3/3 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | reached=3/3 |
| mid | CNN-DDQN | 0.667 | 26.7082 | 18.6250 | reached=2/3, timeout=1/3 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | reached=3/3 |
| long | CNN-DDQN | 0.333 | 47.5907 | 40.4500 | reached=1/3, timeout=2/3 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | reached=3/3 |

结论：该“诊断候选”在随机分布下**显著退化 SR（mid/long 超时）**，未通过 smoke 门；不作为 V8 推进候选。

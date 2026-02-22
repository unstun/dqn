# v8p1 结果对比（infer-only 与 train+infer smoke 已回填）

## 1. 数据来源

### 1) infer-only smoke（固定 v7p1 checkpoint）
- `grid4`：`runs/v8p1_navdist_infer_smoke/20260223_021151/table2_kpis_mean_raw.csv`
- `euclid`（对照）：`runs/v8p1_navdist_infer_smoke/20260223_021220/table2_kpis_mean_raw.csv`

### 2) train+infer smoke（episodes=150, runs=3）
- train：`runs/v8p1_navdist_smoke/train_20260223_021339`
- infer：`runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932/table2_kpis_mean_raw.csv`

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地/远端均已通过）

## 3. short/mid/long 指标（infer-only，固定 v7p1 checkpoint）

> 说明：本节为 infer-only 对照（不训练）；train+infer smoke 结果另见第 1.2 与后续回填。

### 3.1 `grid4`（v8 目标口径）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 16.0346 | 10.0000 | 0.110611 | 0.088 | 0.088 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 0.667 | 25.4944 | 15.6750 | 0.215330 | 0.090 | 0.090 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 0.667 | 47.7609 | 27.0750 | 0.197133 | 0.260 | 0.260 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

### 3.2 `euclid`（对照）
- 仅列 CNN-DDQN 关键 KPI（同一固定 checkpoint）：
  - short：`SR=1.000`，`L=16.2020`，`T=10.3333`
  - mid：`SR=1.000`，`L=24.9075`，`T=15.9500`
  - long：`SR=1.000`，`L=54.7787`，`T=37.4000`

## 4. 门槛检查（infer-only，短/长 hard gate；以 `grid4` 口径计算）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 是（1.000 = 1.000） | 是（16.0346 < 17.0342） | 是（10.0000 < 10.2667） | 通过 |
| long | 否（0.667 < 1.000） | 否（47.7609 > 43.0107） | 否（27.0750 > 22.8167） | 不通过 |

## 5. 结论（infer-only，go/no-go）
- 结论：`NO-GO（infer-only 直接启用 grid4 会掉 SR）`
- 现象：
  - `grid4` 相对 `euclid`：long 的 `avg_path_length/path_time_s` 明显下降，但 mid/long 出现 `collision`，导致 `success_rate` 从 `1.0` 掉到 `0.667`。
- 下一步：
  - `train+infer smoke` 已完成（见第 6～9 节）；结论仍为 `NO-GO`，下一轮按 `v8p2` 方向继续消融与修正。

## 6. short/mid/long 指标（train+infer smoke，episodes=150, runs=3）

> 说明：本节使用 v8p1 训练产物的 smoke 推理结果（代表 run 见第 1.2）。

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 15.3406 | 10.6500 | 0.200831 | 0.362 | 0.362 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 0.333 | 23.0170 | 14.2000 | 0.078648 | 0.253 | 0.253 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 0.333 | 61.9724 | 42.4000 | 0.331291 | 0.446 | 0.446 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

## 7. `failure_reason` 分布（train+infer smoke，CNN-DDQN，n=9）

- reached：4
- timeout：4
- collision：1

## 8. 门槛检查（train+infer smoke，短/长 hard gate）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 否（0.667 < 1.000） | 是（15.3406 < 17.0342） | 否（10.6500 > 10.2667） | 不通过 |
| long | 否（0.333 < 1.000） | 否（61.9724 > 43.0107） | 否（42.4000 > 22.8167） | 不通过 |

## 9. 结论（train+infer smoke，go/no-go）

- 结论：`NO-GO（smoke SR 明显退化）`
- 现象：
  - 虽然 short 的 `avg_path_length` 略优于 baseline，但 `success_rate` 与 `path_time_s` 未同时满足 hard gate。
  - mid/long 以 `timeout` 为主，且 long 的 `avg_path_length/path_time_s` 明显劣于 baseline。
- 下一步（v8p2 候选）：
  - 排查 `grid4_goal_dist_m`（grid BFS 距离场）的插值数值稳定性（尤其是 `inf`/不可达区域参与插值的情况），避免 `NaN` 传导到 reward/gating。
  - 做“解耦消融”：保留 `grid4` 仅用于 reward progress，但推理期 admissible gating / fallback 仍用 `euclid`，先把 SR 拉回 `≈1.0` 再谈 path/time 的收益。

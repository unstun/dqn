# v8p1 结果对比（infer-only 已回填；train+infer 待回填）

## 1. 数据来源

### 1) infer-only smoke（固定 v7p1 checkpoint）
- `grid4`：`runs/v8p1_navdist_infer_smoke/20260223_021151/table2_kpis_mean_raw.csv`
- `euclid`（对照）：`runs/v8p1_navdist_infer_smoke/20260223_021220/table2_kpis_mean_raw.csv`

### 2) train+infer smoke（episodes=150, runs=3）
- train：`N/A`
- infer：`N/A`

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
  - 继续完成 `train+infer smoke`（v8p1 训练侧对齐 progress 距离），检查 SR 是否能恢复到 `≈1.0`，并评估 `avg_path_length/path_time_s` 是否保留收益。

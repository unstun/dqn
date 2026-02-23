# v8p2 结果对比（smoke）

## 1. 数据来源

### 1) infer-only smoke（固定 v7p1 checkpoint，runs=3）

- 固定模型（checkpoint）：`runs/v7p1_train300_esbest/train_20260221_010743`
- D0：`dijkstra8_nocorner`（`w_clearance=2.0`，`sigma_m=0.5`）
  - `runs/v8p2_costmap_infer_smoke/20260223_104100/table2_kpis_mean_raw.csv`
- E0：`euclid`（对照）
  - `runs/v8p2_costmap_infer_smoke/20260223_104135/table2_kpis_mean_raw.csv`
- D1：`dijkstra8_nocorner`（消融：`w_clearance=0.0`，`sigma_m=0.5`）
  - `runs/v8p2_costmap_infer_smoke/20260223_104209/table2_kpis_mean_raw.csv`

### 2) train+infer smoke（episodes=150, runs=3）
- train：`runs/v8p2_costmap_smoke/train_20260223_104408`
- infer（seed=33）：`runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027/table2_kpis_mean_raw.csv`
- infer（seed=34 复测）：`runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608/table2_kpis_mean_raw.csv`

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地）

### 单元测试
- `conda run -n ros2py310 python -m pytest -q`
- 结果：`PASS`（本地，`25 passed`）

## 3. short/mid/long 指标（infer-only，固定 v7p1 checkpoint）

> 说明：infer-only 用于隔离“仅替换 progress 距离（reward/gating/fallback）”的收益上限；三组 baseline（Hybrid A*-MPC）一致，便于横向对比。

### CNN-DDQN（runs=3，mean）

| 组别 | mode | `w_clearance` | `sigma_m` | short SR | short L | short T | mid SR | mid L | mid T | long SR | long L | long T | long inad | long fb |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | `euclid` | N/A | N/A | 1.000 | 16.2020 | 10.3333 | 1.000 | 24.9075 | 15.9500 | 1.000 | 54.7787 | 37.4000 | 0.303 | 0.303 |
| D0 | `dijkstra8_nocorner` | 2.0 | 0.5 | 1.000 | 15.9569 | 9.9667 | 1.000 | 25.0974 | 15.9500 | 1.000 | 52.5492 | 31.7333 | 0.257 | 0.257 |
| D1 | `dijkstra8_nocorner` | 0.0 | 0.5 | 1.000 | 17.0206 | 11.6167 | 1.000 | 25.0347 | 15.9167 | 1.000 | 49.1883 | 31.3000 | 0.238 | 0.238 |

注：
- `L` = `avg_path_length`（越小越好）
- `T` = `path_time_s`（越小越好）
- `inad` = `argmax_inadmissible_rate`
- `fb` = `fallback_rate`

## 4. short/mid/long 指标（train+infer smoke）

### CNN-DDQN（runs=3，mean）

| infer | seed | short SR | short L | short T | mid SR | mid L | mid T | long SR | long L | long T | long inad | long fb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0 | 33 | 0.667 | 15.1498 | 10.2250 | 1.000 | 25.9704 | 17.5000 | 1.000 | 49.1593 | 29.3667 | 0.255 | 0.255 |
| I1 | 34 | 0.667 | 18.4391 | 11.7000 | 1.000 | 28.5336 | 18.7000 | 1.000 | 50.0228 | 29.8667 | 0.248 | 0.248 |

### `failure_reason` 分布（CNN-DDQN）
- I0（seed=33）：short=`reached=2, collision=1`；mid=`reached=3`；long=`reached=3`
- I1（seed=34）：short=`reached=2, collision=1`；mid=`reached=3`；long=`reached=3`

## 5. 门槛检查（smoke）
- infer-only：E0/D0/D1 均 `SR=1.0`（通过 smoke 的 SR gate；long 的 `L/T` 有明显改善）。
- train+infer：mid/long `SR=1.0`，但 short 复现 `SR=0.667（collision=1/3）` → 不满足“`SR≈1.0` 前提下压 `avg_path_length/path_time_s`”的约束。

## 6. 结论（go/no-go）
- 结论：`NO-GO`（暂不进入 full `runs=20`）。
- 已验证：`dijkstra8_nocorner` 能在固定 checkpoint 下把 long `path_time_s 37.4000 -> 31.3~31.7`，且 `SR=1.0`。
- 主要问题：训练后 short 套件出现可复现的 `collision`（两次 seed 复测一致）；下一轮优先解决 short 安全性后再考虑 full。

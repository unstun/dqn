# v8p2 版本说明（Costmap Dijkstra progress distance）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p1`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**已跑 smoke（mid/long SR=1.0；short 出现 collision）**

## 本版目标
- 在 `success_rate≈1.0` 前提下（优先 long），压 `avg_path_length` / `path_time_s`。
- 保持评测分布稳定：random start/goal 的距离筛选与 short/mid/long 套件划分仍沿用旧欧氏距离口径（避免分布漂移）。

## 方法摘要

### 1) 新增 progress 距离模式：`dijkstra8_nocorner`
- 配置键：`forest_progress_dist_mode`（progress 距离口径开关）
  - `euclid`：欧氏距离（旧行为）
  - `grid4`：4 邻接 BFS shortest-path（v8p1）
  - `dijkstra8_nocorner`：8 邻接 Dijkstra + 禁止穿角（v8p2）

### 2) costmap 代价（clearance 惩罚）
- 新增参数：
  - `forest_progress_cost_w_clearance`（clearance 惩罚权重）
  - `forest_progress_cost_sigma_m`（惩罚衰减长度，单位 m）
- 目标：让 progress 距离不仅 obstacle-aware，还对“贴障碍的风险路径”更敏感，以减少 timeout/碰撞并提升 long 的整体效率。

### 3) 数值稳定性：finite-safe progress 采样
- 对含 `inf` 的 progress 距离场，采样使用 finite-safe 双线性插值，避免 `inf*0 -> NaN` 传导污染 reward/gating/fallback。

## 本轮关键命令（计划执行）

### 1) infer-only smoke（固定 v7p1 checkpoint，对照）
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-dist-mode euclid`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-cost-w-clearance 0`（消融：关闭 clearance 代价）

### 2) train+infer smoke（episodes=150, runs=3）
- `conda run -n ros2py310 python train.py --profile repro_20260223_v8p2_costmap_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_smoke --seed 34`（复测）

## 代表 run（已回填）
- infer-only（固定 v7p1 checkpoint）：
  - `dijkstra8_nocorner`（w=2.0）：`runs/v8p2_costmap_infer_smoke/20260223_104100`
  - `euclid`（对照）：`runs/v8p2_costmap_infer_smoke/20260223_104135`
  - `dijkstra8_nocorner`（w=0.0 消融）：`runs/v8p2_costmap_infer_smoke/20260223_104209`
- train：`runs/v8p2_costmap_smoke/train_20260223_104408`
- infer：
  - seed=33：`runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027`
  - seed=34：`runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110608`

## 结论（smoke）
- infer-only：在固定 `v7p1` checkpoint 下，`dijkstra8_nocorner` 能在保持 `SR=1.0` 的前提下显著压 long 的 `avg_path_length/path_time_s`（相对 euclid 对照）。
- train+infer：mid/long `SR=1.0` 且 long `path_time_s` 明显下降，但 short 复现 `collision=1/3`（两次 seed 复测一致）→ 暂不进入 full。

# v8p1 版本说明（NavDist progress distance）

- 版本类型：**Major（v+1）**
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（待 smoke）**

## 本版目标
- 在 `success_rate≈1.0` 前提下（优先 long），压 `avg_path_length` / `path_time_s`。
- 保持评测分布稳定：random start/goal 的距离筛选与 short/mid/long 套件划分仍沿用旧的欧氏距离口径（避免分布漂移）。

## 方法摘要
- 新增 `forest_progress_dist_mode`（progress 距离口径开关）：
  - `euclid`（默认）：旧行为，欧氏距离场作为 progress
  - `grid4`：基于 `traversable_base` 的 4 邻接 BFS 最短路距离（obstacle-aware）
- `grid4` 用途：
  - reward 的 progress 项（`dist_before - dist_after`）
  - 推理期 admissible gating 的进度判据（`min_progress_m`）
  - fallback 动作选择（短 horizon rollout 后选最小 dist）

## 本轮关键命令（计划执行）

### 1) infer-only smoke（固定 v7p1 checkpoint，对照 grid4 vs euclid）
- `conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_infer_smoke --forest-progress-dist-mode euclid`

### 2) train+infer smoke（episodes=150, runs=3）
- `conda run -n ros2py310 python train.py --profile repro_20260222_v8p1_navdist_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260222_v8p1_navdist_smoke`

## 代表 run（待回填）
- infer-only：`N/A`
- train：`N/A`
- infer：`N/A`

## 结论（待回填）
- `N/A`


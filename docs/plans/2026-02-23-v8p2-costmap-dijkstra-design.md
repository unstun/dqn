# V8p2 设计草案：Costmap Dijkstra progress distance（dijkstra8_nocorner）用于 reward + admissible gating + fallback

日期：2026-02-23  
作者：Codex（协作）

## 1. 背景与目标

### 1.1 背景
- `v8p1` 已引入 `forest_progress_dist_mode=grid4`（4 邻接 BFS 最短路距离）作为 progress（进度）距离，用于：
  - reward progress（`dist_before - dist_after`）
  - 推理期 admissible gating（`min_progress_m` 判据）
  - fallback（短 rollout 后选最小 dist）
- 但 `v8p1` 的 train+infer smoke 结果显示 SR 明显退化，失败以 `timeout` 为主（mid/long 尤其明显），说明“仅换成 obstacle-aware shortest-path（等权）”仍不足以稳定收敛/推理行为。

### 1.2 v8p2 目标（硬约束口径）
- 首要目标：在 `success_rate≈1.0` 前提下（优先 long），压 `avg_path_length` / `path_time_s`。
- 评测分布稳定性：random start/goal 的距离筛选与 short/mid/long 套件划分仍沿用旧欧氏口径（避免分布漂移）。

## 2. 方案概述（选择 C：加权 Dijkstra / costmap distance）

### 2.1 核心思路
- 将 progress 距离从“等权 shortest-path（BFS）”升级为“带障碍风险权重的最短路代价场”：
  - 在 `traversable_base`（基于 EDT + footprint clearance 的可通行集合）上构图；
  - 边代价为 “步长（m）× cell cost multiplier（代价倍率）”；
  - 用 Dijkstra 从 goal 反推到全图，得到 `progress_cost_to_goal_m[y,x]`（单位：m 等价代价）。
- 采用 `dijkstra8_nocorner`：8 邻接 + 禁止穿角（diagonal move 需满足相邻两正交 cell 均可通行），减少“贴角钻缝”的非物理行为。

### 2.2 代价倍率（costmap）定义（最小高置信）
- 基于 EDT 距离 `dist_m`（到最近障碍/边界距离）构造 clearance 相关的倍率：
  - `od_base_m = max(0, dist_m - clearance_thr_m)`（超出“刚好不碰撞”的净空）
  - `penalty = exp(-od_base_m / sigma_m)`（离障碍越近 penalty 越大）
  - `cost_factor = 1 + w_clearance * penalty`（>=1，近障碍更贵）
- 超参：
  - `w_clearance`：clearance 惩罚权重（>=0）
  - `sigma_m`：衰减长度（>0，单位 m）

### 2.3 使用位置（保持 v8 的“对齐 progress”原则）
`dijkstra8_nocorner` 产出的 progress 代价场用于：
1) reward 的 progress 项：`dist_before - dist_after`  
2) 推理期 admissible gating：以该距离判定 `dist0 - dist1 >= min_progress_m`  
3) fallback 动作选择：短 horizon rollout 后选最小 dist  

## 3. 关键工程细节（必须做，否则易出现数值不稳定）

### 3.1 `inf*0 -> NaN` 的插值传导风险
- `grid4`/Dijkstra 的距离场对不可达区域会含 `inf`。
- 若仍使用普通双线性插值（`bilinear_sample_2d_vec`），在角点包含 `inf` 且权重为 0 的情况下会产生 `NaN`（典型：`inf * 0`），进而污染 reward/gating/fallback 判据，导致 timeout/行为异常。
- v8p2 要求：progress 距离采样统一使用 finite-safe 插值（`bilinear_sample_2d_finite(_vec)`），把非有限角点替换为 `fill_value`，保证不产生 `NaN` 传导。

### 3.2 计算与缓存策略
- Dijkstra 只在 goal 确定后计算一次，并按 `(goal_xy, mode)` 缓存；每 step 的 progress 查询只做数组采样，不重复跑 Dijkstra。
- 训练中 goal 会随 episode 重置变化：每 episode 需要 1 次 Dijkstra（对 `forest_a` 360×360，预期可接受；若后续成为瓶颈，再考虑降采样/局部更新）。

## 4. 非目标（明确不做）
- 不改变 random start/goal 的距离筛选口径（仍用欧氏 `_goal_dist_m`）。
- 不引入新依赖（如 `scipy`）。
- 不在本轮引入“reward 与推理期 gating/fallback 解耦”的新开关（先把 v8p2 作为单变量对照跑通 smoke，再决定是否做解耦消融）。

## 5. 验证计划（时间优先）

1) 最小自检：
   - `conda run -n ros2py310 python train.py --self-check`
   - `conda run -n ros2py310 python infer.py --self-check`
2) 单元测试：
   - 覆盖 `dijkstra8_nocorner` 的对角步长、禁穿角、加权代价正确性与不可达=inf；
   - 覆盖 finite-safe 插值不产生 `NaN`。
3) smoke：
   - 远端 `ubuntu-zt`：`episodes=150` + `runs=3`（short/mid/long），与 `v7p1` 同口径对照；
   - 以 SR 为 hard gate，若未恢复 `≈1.0`，不进入 full。

## 6. 联网调研（可追溯）

- 本轮为小版本（`p+1`）工程改动，暂不做额外联网调研；若 v8p2 smoke 仍不稳定，再在进入大改动前补“近两年论文/仓库”调研记录。


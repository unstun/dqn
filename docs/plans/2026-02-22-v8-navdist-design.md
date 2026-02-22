# V8 设计草案：Obstacle-aware progress distance（grid shortest-path）用于 reward + admissible gating

日期：2026-02-22  
作者：Codex（协作）

## 1. 背景与目标

### 1.1 背景
- 当前 `AMRBicycleEnv`（森林车辆环境）在 reward shaping 与推理期 admissible gating（可采纳动作过滤）中使用“到目标的欧氏距离场”（`euclidean_goal_dist_m(...)`）作为 progress（进度）判据。
- 在存在障碍/绕行的场景中，欧氏距离会出现“看起来离目标更近但实际被障碍阻隔”的假进度，导致：
  - reward 的 progress 梯度不稳定（对绕行方向缺少一致引导）；
  - admissible gating 误杀动作（错误判断“无进度”），增加 fallback/override 触发，影响 `path_time_s` 与路径质量。
- 现有推理期消融已表明：仅微调推理期 gating 参数对 `path_time_s` 有一定空间，但对 `avg_path_length` 的系统性改善有限；若要在 `success_rate≈1.0` 前提下压 `avg_path_length/path_time_s`，需要更“根因”的进度定义对齐。

### 1.2 V8 目标（硬约束口径）
- 在 `success_rate≈1.0`（优先 long 套件）前提下，压 `avg_path_length` 与 `path_time_s`（越小越好）。
- 保持评测/采样口径稳定：random start/goal 的距离约束与套件划分仍使用旧口径（避免分布漂移），仅替换 progress 的“距离定义”。

## 2. 方案候选与取舍

### 方案 A（推荐）：grid shortest-path distance（BFS geodesic）作为 progress 距离
- 思路：在离散栅格上，以“可通行集合”（基于现有 clearance 阈值的 `traversable_base`）为图节点，采用 4 邻接 BFS 从 goal 反推到全图，得到 obstacle-aware 的最短路距离场（单位 m）。
- 使用位置：
  1) reward progress：`reward_k_p * (dist_before - dist_after)`  
  2) 推理期 admissible gating：以该距离判定 `dist0 - dist1 >= min_progress_m`
  3) fallback 动作选择（短 horizon rollout 后选最小 dist）
- 优点：实现简单、无新依赖、距离定义与障碍一致；对绕行场景更稳定；可用开关保持默认行为不变。
- 风险：每回合 goal 变化时需重算距离场（训练中 goal 随机化），需要避免在“随机采样 tries”过程中重复重算，保证开销可控。

### 方案 B：使用 Hybrid A* 路径长度（或 A*）作为进度
- 思路：每步/每次判定用规划器计算从当前到 goal 的最短路长度。
- 优点：更贴近真实可行路径。
- 缺点：每步调用规划器开销过大（尤其训练）；实现与缓存复杂；容易污染 `kpi_time_mode=policy` 的计时口径。

### 方案 C：下采样距离场（粗分辨率 geodesic）
- 思路：在较粗网格上做 BFS，映射回连续坐标。
- 优点：更快。
- 缺点：引入额外超参（下采样倍率/插值），在未证明瓶颈前不建议先上复杂度。

结论：先落地方案 A（最小高置信、可开关），若后续发现 compute 成本成为瓶颈，再升级到方案 C。

## 3. 设计细节（推荐方案 A）

### 3.1 新增开关（默认不改变行为）
- 新增参数：`forest_progress_dist_mode`（progress 距离口径）
  - 可选：`euclid`（默认，保持旧行为）、`grid4`（4 邻接 BFS geodesic）
- 入口：
  - CLI：`--forest-progress-dist-mode {euclid,grid4}`
  - JSON profile：`forest_progress_dist_mode: "grid4"`

### 3.2 距离场计算与缓存策略
- 基于现有 `self._traversable_base`（由 EDT + footprint clearance 构造）生成 geodesic 距离：
  - goal cell 距离为 0
  - 遍历可通行 cell，邻接扩张 4 方向
  - 不可达 cell 距离为 `inf`
- 关键约束：**避免在 `_sample_random_start_goal` 的 goal-tries 循环里计算 geodesic**  
  - 现状：该循环会多次调用 `_set_goal_xy(...)` 以测试候选 goal 的欧氏距离约束
  - 方案：`_set_goal_xy(...)` 仍只维护欧氏距离场；geodesic 距离在 reset 最终确定 goal 后一次性计算（或 lazy 计算一次，并按 `(goal_xy, mode)` 缓存）。

### 3.3 使用位置（行为变更点）
- `reward`：
  - 将 progress 的 `dist_before/dist_after` 由欧氏距离切换为 progress 距离（`grid4` 时为 geodesic）
  - 若 progress 距离不可用（`inf`/`nan`），回退到欧氏直线距离 `d_goal_before - d_goal_after`（保持鲁棒性）
- `admissible_action_mask(...)` / `is_action_admissible(...)`：
  - 以 progress 距离判定 `min_progress_m`
  - 其它安全条件（collision/OD、goal relax、reverse unlock）保持不变
- `_fallback_action_short_rollout(...)`：
  - 用 progress 距离选择“最小 dist”动作，避免在绕行区被欧氏距离误导

### 3.4 非目标（明确不做）
- 不改变 random start/goal 的距离筛选口径（仍用 `_goal_dist_m` 欧氏距离场），不改变套件 short/mid/long 的采样方式。
- 不引入新依赖（如 `scipy`），不替换算法主干/网络结构。

## 4. 风险与应对

1) **性能风险**：训练阶段每 episode goal 变化导致 geodesic 重算  
   - 应对：只在最终 goal 确定后计算一次；对 360×360 图单次 BFS 预计可接受；必要时再引入下采样版本。
2) **不可达状态导致 progress=inf**  
   - 应对：在 progress 不有限时回退到欧氏直线距离差；同时保持 `rand_reject_unreachable`（推理）口径。
3) **评测计时口径被污染**（`kpi_time_mode=policy`）  
   - 应对：在 `env.reset(...)` 阶段完成 geodesic 计算，避免在每步 action selection 期间触发计算。

## 5. 验证计划（时间优先）

1) 最小自检：  
   - `conda run -n ros2py310 python train.py --self-check`  
   - `conda run -n ros2py310 python infer.py --self-check`
2) infer-only smoke（固定 v7p1 checkpoint，对照 v7p1 现状）：  
   - `runs=3`，仅切换 `forest_progress_dist_mode={euclid,grid4}`，观察 SR/L/T 与 fallback/inadmissible。
3) train+infer smoke：  
   - `episodes=150`、`runs=3`（short/mid/long），仅在 V8 profile 中启用 `grid4`；其余保持与 v7p1 同口径。
4) 若 smoke 显示收益，再进入 full `runs=20`（short/long 硬门槛口径）。

## 6. 联网调研（可追溯）

检索时间：2026-02-22（本地时间）

- Habitat-Lab 文档：Geodesic distance API（用于 embodied navigation 的最短路距离度量/常用于 success 与 SPL 等指标）
  - https://aihabitat.org/docs/habitat-lab/habitat.utils.visualizations.utils.html#habitat.utils.visualizations.utils.observations_to_image
  - https://aihabitat.org/docs/habitat-lab/habitat.utils.visualizations.utils.html#habitat.utils.visualizations.utils.get_fastest_action_to_goal
- AllenAct 教程：PointNav reward 中显式使用 geodesic distance（并建议预计算最短路距离用于 reward shaping）
  - https://allenact.org/tutorials/training-a-pointnav-model/
- 相关论文（ToA maps / obstacle-aware distance-to-goal 作为导航信号的思想参考）：
  - Velsamas et al., “On the Utility of Time-of-Arrival Maps in Robotic Navigation,” arXiv:2506.03705 (2025)
    - https://arxiv.org/abs/2506.03705


# v8p10 hybrid（SR≈1.0 前提下压 avg_path_length / path_time_s：追平/反超 baseline）

**日期**：2026-02-24  
**背景**：v8p9 的 fixed pairs3 sweep smoke 显示：通过推理侧阈值调整可以把 RL 的 `success_rate` 拉回到 1.0，但 `avg_path_length/path_time_s` 仍明显落后 `Hybrid A*-MPC`（尤其 long）。为了更快闭环，本版优先做“推理侧可解释消融”，先确认导致绕路/变慢的主要因子，再决定是否进入更重的 DQN 结构改动。

## 目标与约束

- 目标（最终 gate C）：short/long 各 `runs=20`（fixed pairs20）下同时满足：
  - `success_rate(RL) >= success_rate(Hybrid A*-MPC)`
  - `avg_path_length(RL) < avg_path_length(Hybrid A*-MPC)`
  - `path_time_s(RL) < path_time_s(Hybrid A*-MPC)`
- 反作弊：不改 `goal_tolerance_m`（终点容差），不通过放宽 stop 判定“蹭过终点”。
- 公平性：baseline 必须同跑；所有结论以 fixed pairs20 + `runs=20` 为准。
- 口径声明：本版允许 `hybrid/shielded` 推理策略（masking/replacement/fallback），但必须在版本留档中明确标注，不得宣称 `strict-argmax`。

## 核心假设（为什么 RL 路径更长）

1) `progress_dist`（进度距离场）若引入了 “靠近障碍更贵” 的代价（`forest_progress_cost_w_clearance>0`），会系统性鼓励“离障更远的路线”，导致平均路长上升。  
2) replacement（当 `argmax(Q)` 不可采纳时的替换动作选择）如果把 clearance（OD）作为高优先级 tie-break，也会把策略推向“更保守更绕”的动作序列。

## 设计思路（推理优先 + 可解释消融）

### 1) progress distance（主轴消融）

- 固定 `forest_progress_dist_mode=dijkstra8_nocorner`（障碍感知最短路 cost-to-go），先扫：
  - `forest_progress_cost_w_clearance ∈ {0.0, 0.5, 1.0, 2.0}`
  - `forest_progress_cost_sigma_m` 固定（默认 0.5）
- 预期：`w_clearance=0` 更接近“几何最短”，有利于路长；安全性由 `min_od_m` 与短视 rollout 兜底保证。

### 2) replacement ranking（必要时的最小代码改动）

在现有 `forest_replace_ranking`（`q`/`progress_clearance_q`/`clearance_progress_q`）基础上新增：
- `progress_q`：先最小化 next-step `progress_dist`，再最大化 Q（最后不看 clearance）
- `progress_q_clearance`：先最小化 next-step `progress_dist`，再最大化 Q，最后最大化 clearance（OD）

目标：在候选动作均安全（coll-free）时，减少“因 clearance 优先导致的绕行”，同时保留 Q 的长期回报偏好。

## 评测口径

- sweep smoke：short/long 各固定 pairs3（从 pairs20 子集抽取）+ `runs=3`，用于快速排序与定位主因。
- full gate：short/long fixed pairs20 + `runs=20`，作为唯一最终裁决。


# v8p9 推理侧 sweep 设计（SR≈1.0 前提下压 avg_path_length / path_time_s）

**日期**：2026-02-24  
**背景**：v8p8 smoke（runs=3）显示 mid/long 的 `avg_path_length/path_time_s` 明显落后 `Hybrid A*-MPC`，且 short SR 仍低于 baseline。为最快闭环，本版先对推理侧阈值做可复现 sweep，确认“推理口径上限”，再决定是否进入训练侧（reward / DQfD 约束）或更重的 DQN 变种。

## 目标与约束

- 目标（最终 gate C）：short/long 各 `runs=20`（fixed pairs20）下同时满足：
  - `success_rate(RL) >= success_rate(Hybrid A*-MPC)`
  - `avg_path_length(RL) < avg_path_length(Hybrid A*-MPC)`
  - `path_time_s(RL) < path_time_s(Hybrid A*-MPC)`
- 反作弊：不改 `goal_tolerance_m`（终点容差）；默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。
- 公平性：所有对比使用 fixed pairs（避免 sample drift）；baseline 必须同跑。

## 设计思路（推理优先）

推理时 RL 策略的动作“可采纳（admissible）”由短视 rollout 的安全与进度共同决定，相关阈值直接影响：
- 可采纳动作集大小（过严会筛空，引发 fallback；过松会走险/抖动）
- 替换动作（replacement）的偏好（可能引入绕路/停滞）
- 近目标阶段的速度整形（影响 stop 成功与时间）

因此，优先 sweep 以下参数（短/长分开跑）：
- `forest_min_progress_m`：提高可减少抖动、缩短时间，但可能增加 fallback 或卡死
- `forest_min_od_m`：降低可能缩短路径，但 SR 风险上升
- `forest_replace_topq`：越小越接近纯 Q（更快，但可能更危险）
- `forest_topk_turn_penalty`：抑制急拐与 zigzag（潜在缩短路径与时间）
- `forest_goal_approach_speed_factor`：提高可缩短时间，但可能影响 stop 成功

## 评测口径

- sweep smoke：short/long 各固定 pairs3（从 pairs20 子集抽取）+ `runs=3`，用于快速排序
- full gate：short/long fixed pairs20 + `runs=20`，作为唯一最终裁决


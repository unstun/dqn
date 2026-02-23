# v8p3 版本说明（collision-first fallback safety）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p2`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**smoke 失败（mid collision=1/3；long timeout=1/3）**

## 本版目标（smoke 口径）

硬约束（优先）：
- 在 short/mid/long smoke 下尽量保持 `success_rate≈1.0`，并优先消除 short 可复现 `collision`。

次目标：
- 在不牺牲 SR 的前提下继续压 `avg_path_length` / `path_time_s`（延续 v8p2 的 long 收益）。

## 方法摘要

### 1) 问题：mask 为空时可能落入“允许碰撞”的最后兜底

在 `v8p2` 的 train+infer smoke 中，short 出现可复现 `collision=1/3`。初步定位为：
- progress mask 为空 → 回退 safe mask；
- safe mask 仍强制 `min_od_m`（最小净空阈值/clearance 下限）；
- 在狭窄处可能出现“存在 collision-free 动作但全部 `min_od < min_od_m`”→ safe mask 仍为空；
- 随后进入 `_fallback_action_short_rollout(...)` 的最后兜底（注释写明可能返回碰撞动作）。

### 2) 修复：collision-first 的 fallback（最小行为修复）

修改 `AMRBicycleEnv.admissible_action_mask(..., fallback_to_safe=True)`（可行动作掩码生成）：
1) 若 `out` 为空，优先回退到 `safe=(~coll) & (min_od>=min_od_m)`（保持原语义）；
2) 若 `safe` 仍为空但存在 collision-free，则再回退到 `(~coll)`（放松 `min_od_m`，collision-first）。

说明：本版不改动超参/奖励/采样分布，尽量保持单变量消融清晰。

## 本轮关键命令（计划执行）

### 1) smoke（episodes=150, runs=3）
- `conda run -n ros2py310 python train.py --profile repro_20260223_v8p3_fallback_safety_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_fallback_safety_smoke`

### 2) 回归（复现 v8p2 short collision pair，infer-only）
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p3_short_collision_pair_regression`

## 代表 run（待回填）
- smoke：`runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153`
- 回归：
  - shielded/hybrid：`runs/v8p3_short_collision_pair_regression/20260223_124513`
  - strict-argmax（诊断）：`runs/v8p3_short_collision_pair_regression_strict/20260223_124959`

## 结论（待回填）
- 结论：`NO-GO`（mid collision + long timeout），不进入 full；按工作流进入 `v8p4` 继续迭代（更强 safety 兜底/屏蔽）。

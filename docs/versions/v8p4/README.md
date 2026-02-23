# v8p4 版本说明（short-rollout 的 1-step collision-free 降阶兜底）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p3`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**回归失败（collision + timeout 仍存在；暂不进入 smoke）**

## 本版目标（smoke 口径）

硬约束（优先）：
- 在 short/mid/long smoke 下尽量保持 `success_rate≈1.0`，并优先消除 `v8p3` smoke 的 `mid collision` 与 `long timeout`。

次目标：
- 在不牺牲 SR 的前提下继续压 `avg_path_length` / `path_time_s`（延续 `v8p2` 的 long 收益）。

## 方法摘要

### 1) 问题：最后兜底可能返回“立即碰撞”动作

当推理阶段多层 admissible gating 都失败（top-k / mask 为空）时，会进入 `_fallback_action_short_rollout(...)`（短视安全兜底）。  
该函数原逻辑在 “多步 horizon 下找不到 safe” 时，会落到一个仅按 “1-step 最大净空” 的选择分支（注释也允许碰撞）。在狭窄区域，这可能导致：

- `adm_horizon=30` 下所有“恒定动作”都在 30 步内碰撞（`coll=True`），导致 safe 集合为空；
- 但存在 `h=1` 的 collision-free 动作（只是“未来会撞”，并非“立即撞”）；
- 若最后兜底仍可能选择“立即碰撞”动作，就会把 SR 拉爆（collision 回潮）。

### 2) 修复：h=1 collision-first 的降阶兜底（不改超参，便于消融）

修改 `_fallback_action_short_rollout(horizon_steps=h)` 的分支优先级：

1. 先按原逻辑：在 `h` 步 rollout 下找 `(~coll) & (min_od>=min_od_m)` 的动作（progress-first）。
2. 若为空：额外尝试 `h=1` rollout，只要存在 `~coll`（1-step collision-free），就只在这些动作中选择 `progress_dist`（进展距离）最小者，`min_od`（净空）打破平局。
3. 仅当所有动作在 `h=1` 下也都碰撞时，才允许进入“clearance-only（可能碰撞）”的最后兜底。

直觉：当 multi-step horizon 过严导致“未来不可行”时，至少先保证“下一步不撞”（collision-first）。

## 本轮关键命令（计划执行）

### 1) 回归（v8p3 smoke failures，infer-only）
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_smoke_failures_regression`
- pairs：`configs/pairs_v8p3_smoke_failures.json`

### 2) smoke（episodes=150, runs=3）
- `conda run -n ros2py310 python train.py --profile repro_20260223_v8p4_fallback_h1_smoke`
- `conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_fallback_h1_smoke`

## 代表 run

- 回归：`runs/v8p4_smoke_failures_regression/20260223_142739`
- smoke：`N/A`

## 结论

- 当前结论：`NO-GO`（fixed pairs 回归仍出现 `collision/timeout`；下一步进入 `v8p5` 做更强的 safety/shield 迭代后再复测）

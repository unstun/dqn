# v8p7 版本说明（goal-approach speed shaping：接近目标阶段速度整形）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p6`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**infer-only smoke 已通过（seed=33；runs=3；短/中/长 SR=1.0）；train+infer smoke：N/A**

## 本版目标

硬约束：
- 在 `SR≈1.0` 前提下继续压 `avg_path_length` / `path_time_s`，并始终与 `Hybrid A*-MPC`（基线）对比。

本版直接动机（来自 v8p6 训练产物回归定位）：
- short/mid 的 `collision` 多发生在接近 goal 的末段：进入“必撞态”（admissible safe action mask 为空）后再执行 stop/摆正会触发碰撞。
- 因此优先尝试“更早减速/更早调整”而不是到 goal pose 后再 stop override。

## 方法摘要

新增推理侧开关 `--forest-goal-approach-override`（接近目标速度整形）：

- 触发：当 `d_goal_m <= approach_dist`（默认 `2.5 * goal_tolerance_m`）且尚未满足 `reached_stop` 时；
- 行为：保持当前策略选出的 `delta_dot`（转向率）不变，仅在“同一 `delta_dot` 的动作子集”内挑选**admissible** 的 `accel`（加速度），使下一步速度 `|v_next|` 接近保守包络：
  - `v_target = factor * sqrt(v_stop^2 + 2*a_max*(d_goal - goal_tol))`
  - `factor` 由 `--forest-goal-approach-speed-factor` 控制（默认 `0.8`，越小越早刹车）
- 约束：当启用 `--forest-no-fallback`（严格 `argmax(Q)` 口径）时该整形**被忽略**，避免污染 strict 诊断口径。

## 本轮关键命令（计划执行）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) infer-only smoke（固定 `v8p6` 训练产物 checkpoint；runs=3）

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p7_goal_approach_infer_smoke
```

### 3) （待跑）train+infer smoke（episodes=150, runs=3）

```bash
conda run -n ros2py310 python train.py --profile v8p7
conda run -n ros2py310 python infer.py --profile v8p7
```

## 代表 run

- infer-only smoke（fixed models + baseline 对比）：
  - `runs/v8p7_goal_approach_infer_smoke/20260223_230524`

## 结论（本轮已回填）

- 推理侧“接近目标速度整形”在该随机分布 smoke 样本上把 short/mid 的末段碰撞风险压回（短/中/长均 `SR=1.0`）。
- 但 mid/long 的 `path_time_s` 仍显著落后 `Hybrid A*-MPC`（见 `RESULTS.md`），下一步需要围绕 `speed_factor/approach_dist` 做小网格调参（smoke 优先），在不牺牲 SR 的前提下压时间。

下一步（优先级）：
1) `--forest-goal-approach-speed-factor` 由 `0.8 -> 0.9/1.0` 小步上调，观察 `path_time_s` 是否回落且不引入 collision。
2) 视情况缩小 `--forest-goal-approach-dist-m`，避免过早刹车导致 mid/long 变慢。


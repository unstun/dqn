# v8p14 版本说明（推理侧回归：`w_clearance=1.5` + 放松 `min_progress` 以恢复 long 的 SR）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p13`（smoke NO-GO）
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（infer-only sweep，待结果）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；也不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。

## 方法摘要（本版主线）

动机：
- v8p11 long：在 `SR=1.0` 下仍明显 detour（路径/时间落后 baseline）。
- 推理侧 sweep（v8p11 long）：`forest_progress_cost_w_clearance=1.5` 更短，但 `SR` 掉到 `0.667`（timeout），怀疑是 admissible gate 过“刚性”，不允许短视 detour。

v8p14 的做法（infer-only，最快闭环）：
- 固定 `w_clearance=1.5`，并将 `forest_min_progress_m`（可行动作的进度阈值，允许短视回退）设为小负数（如 `-0.02/-0.05`），目标是在不改 goal 判定阈值的前提下恢复 `SR≈1.0`，并尽量保持更短路径倾向。
- 本版 **不训练新模型**：推理加载 `v8p11` 的权重（`models=v8p11`），只改推理侧 gate/距离场参数，便于快速定位瓶颈。

## 本轮关键命令

### 1) 最小自检

```bash
conda run -n ros2py310 python infer.py --self-check
```

### 2) infer sweep（long，fixed pairs3，runs=3，baseline 同跑）

```bash
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg002
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg005
```


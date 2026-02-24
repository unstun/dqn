# v8p15 版本说明（infer-only：sweep `progress_cost_sigma_m`，主攻 long detour）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p14`（infer-only sweep NO-GO）
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**infer-only sweep 已跑（NO-GO；不建议 full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；也不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。

## 方法摘要（本版主线）

v8p14 已验证：在 `w_clearance=1.5` 下把 `forest_min_progress_m` 放松为负数（允许短视 detour）可恢复 long 的 `SR=1.0`，但 long 的 `avg_path_length/path_time_s` 仍显著落后 baseline。

v8p15 固定：
- `forest_progress_cost_w_clearance=1.5`
- `forest_min_progress_m=-0.05`

仅 sweep：
- `forest_progress_cost_sigma_m`（clearance 惩罚衰减长度，影响“偏离障碍物的偏好”在空间上的扩散范围）

本版仍为 infer-only：推理加载 `v8p11` 权重（`models=v8p11`），只改推理侧距离场参数以最快闭环定位瓶颈。

本轮 sweep 结论（fixed pairs3，runs=3）：
- `sigma=0.2` 出现明显绕路（path 爆炸），直接淘汰。
- `sigma=0.3` 为本轮最优（在 SR=1.0 下进一步缩短 long 路径/时间），但相比 baseline 仍明显落后（未达目标），判定 **NO-GO**。

## 本轮关键命令

```bash
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p2
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p3
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p5
```

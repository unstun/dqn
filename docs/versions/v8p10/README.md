# v8p10 版本说明（hybrid 口径下压路径/时间：progress-dist clearance 消融优先）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p9`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（推理侧 sweep smoke → full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值），避免隐性放宽到达判定。

## 方法摘要（本版主线）

本版优先做“可解释推理侧消融”，定位导致绕路/变慢的主因：

- 固定 `forest_progress_dist_mode=dijkstra8_nocorner`（障碍感知最短路 cost-to-go）
- sweep `forest_progress_cost_w_clearance`（靠近障碍的代价权重）：
  - 预期：`w_clearance=0` 更接近几何最短路径；安全性由 `min_od_m` + rollout 兜底保证
- 若仍明显落后 baseline，再进入“最小代码改动”的 replacement ranking（progress-first tie-break）

## 本轮关键命令（计划）

### 1) 最小自检

```bash
conda run -n ros2py310 python infer.py --self-check
```

### 2) 推理侧 sweep smoke（fixed pairs3，runs=3）

> 说明：infer 的 `--rand-pairs-json` 不能同时对 short/long 指定两份文件，因此 short/long 必须分别跑。

```bash
# short
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p10_infer_sweep_short_smoke

# long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p10_infer_sweep_long_smoke
```

### 3) full gate（C：short/long，各 runs=20，fixed pairs20）

```bash
# short
conda run -n ros2py310 python infer.py --profile v8p10 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p10_full20_pairs_short

# long
conda run -n ros2py310 python infer.py --profile v8p10 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p10_full20_pairs_long
```

## 代表 run

- sweep smoke：`N/A`
- full20（pairs20）：`N/A`

## 结论（待回填）

- `N/A`


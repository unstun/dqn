# v8p12 版本说明（对齐 shortest-path 进度距离场：`w_clearance=0`，主攻 long detour）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p11`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（train+infer smoke → 决定是否 full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。

## 方法摘要（本版主线）

v8p11 smoke 显示：short 已可在 SR=1.0 下压过 baseline，但 long 仍存在明显 detour（路径/时间落后）。推理侧 sweep 也表明：仅调 `forest_progress_cost_w_clearance` 不足以把 long 压到 baseline 以下，且更激进的点会掉 SR。

因此 v8p12 优先做“口径对齐”而不是继续推理侧扫参：

- 将 `forest_progress_cost_w_clearance=0.0`（在 `progress_dist_mode=dijkstra8_nocorner` 下）用于 train+infer：
  - 进度距离场回到“更接近几何最短路”的 cost-to-go；
  - 期望减少 long 上的系统性绕路。
- 将 demo 侧 `forest_demo_w_clearance=0.0`：
  - 减少“专家（hybrid planner）采样路径的 clearance 偏好”与“训练期进度奖励”之间的口径冲突。

## 本轮关键命令（计划）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) train smoke（episodes=150）

```bash
conda run -n ros2py310 python train.py --profile repro_20260224_v8p12_train_smoke
```

### 3) infer smoke（fixed pairs3，runs=3，baseline 同跑）

```bash
# short
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p12_infer_smoke_short

# long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p12_infer_smoke_long
```

### 4) full gate（C：short/long，各 runs=20，fixed pairs20）

```bash
# short
conda run -n ros2py310 python infer.py --profile v8p12 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p12_full20_pairs_short

# long
conda run -n ros2py310 python infer.py --profile v8p12 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p12_full20_pairs_long
```

## 代表 run

- train smoke：`N/A`
- infer smoke：`N/A`
- full20（pairs20）：`N/A`

## 结论（待回填）

- `N/A`


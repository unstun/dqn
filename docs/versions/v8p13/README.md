# v8p13 版本说明（训练侧 reward 塑形可调：`k_p/k_o/k_v`，主攻 long detour）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p12`（smoke NO-GO）
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（待跑：train+infer smoke → 决定是否继续 sweep / 进入 full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。

## 方法摘要（本版主线）

v8p11 smoke 显示：short 已可在 `SR=1.0` 下压过 baseline，但 long 仍存在明显 detour（路径/时间落后）。v8p12 尝试把 `progress_dist` 对齐到 shortest-path（`forest_progress_cost_w_clearance=0`）后，long 仅小幅回落且 short 明显回退，判定 NO-GO。

因此 v8p13 转向“训练侧 reward 塑形”来压 long detour：在不改 goal 判定阈值前提下，将以下奖励项暴露为可调超参：

- `forest_reward_k_p`（`k_p`：进度奖励系数）
- `forest_reward_k_o`（`k_o`：近障惩罚系数）
- `forest_reward_k_v`（`k_v`：近障速度耦合惩罚系数）

直觉假设：`k_o/k_v` 过强会诱导策略为了更大 clearance 而绕路；适度降低可在 shielded/hybrid 兜底下缩短 long 路径，同时不显著损害 SR。

## 本轮关键命令（计划）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) train smoke（episodes=150）

```bash
conda run -n ros2py310 python train.py --profile repro_20260224_v8p13_train_smoke
```

### 3) infer smoke（fixed pairs3，runs=3，baseline 同跑）

```bash
# short
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p13_infer_smoke_short

# long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p13_infer_smoke_long
```

### 4) full gate（C：short/long，各 runs=20，fixed pairs20）

> 仅当 smoke 显示明确收益（尤其 long 在 `SR≈1.0` 下明显逼近/超过 baseline），才进入 full gate。

```bash
conda run -n ros2py310 python infer.py --profile v8p13 \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --envs forest_a::short --runs 20 --out v8p13_full_short_pairs20

conda run -n ros2py310 python infer.py --profile v8p13 \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --envs forest_a::long --runs 20 --out v8p13_full_long_pairs20
```


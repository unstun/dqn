# v8p11 版本说明（训练优先：专家+DQfD 强化以压路径/时间；推理口径继承 v8p10）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p10`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（train+infer smoke → full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值）。

## 方法摘要（本版主线）

v8p10 的推理侧消融表明：即使 SR=1.0，RL 的 `avg_path_length/path_time_s` 仍明显落后 baseline（尤其 long）。因此 v8p11 将主攻训练侧，把 baseline 的“短路径偏好”更强地灌进 Q：

- 推理侧口径：继承 v8p10（`dijkstra8_nocorner` + `progress_cost_w_clearance=2.0` + `replace_ranking=progress_q` + `replace_topq=3`）
- 训练侧强化：
  - `forest_expert_exploration=true`（专家混入行为策略）
  - DQfD demo 阶段更强（prefill/pretrain + margin loss 权重上调）

## 本轮关键命令（计划）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) train smoke（episodes=150）

```bash
conda run -n ros2py310 python train.py --profile repro_20260224_v8p11_train_smoke
```

### 3) infer smoke（fixed pairs3，runs=3，baseline 同跑）

```bash
# short
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p11_infer_smoke_short

# long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p11_infer_smoke_long
```

### 4) full gate（C：short/long，各 runs=20，fixed pairs20）

```bash
# short
conda run -n ros2py310 python infer.py --profile v8p11 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p11_full20_pairs_short

# long
conda run -n ros2py310 python infer.py --profile v8p11 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p11_full20_pairs_long
```

## 代表 run

- train smoke：`runs/v8p11/train_20260224_151042`
- infer smoke：
  - short（fixed pairs3）：`runs/v8p11_infer_smoke_short_pairs3/20260224_152858`
  - long（fixed pairs3）：`runs/v8p11_infer_smoke_long_pairs3/20260224_152917`
- full20（pairs20）：`N/A`

## 结论（待回填）

- smoke（runs=3，fixed pairs3）结论：
  - short：SR=1.0，且 `avg_path_length/path_time_s` **均优于** baseline（Hybrid A*-MPC）
  - long：SR=1.0，但 `avg_path_length/path_time_s` **仍劣于** baseline（约 +5.83m / +3.08s）
- 下一步优先级：继续主攻 long（不建议直接上 full gate C），优先做“训练侧/推理侧”针对 long 的消融与再扫参，目标是把 long 的 detour 压到 baseline 以内。

补充（推理 sweep，fixed pairs3 / runs=3）：
- long 的 `forest_progress_cost_w_clearance` sweep 显示：`w=2.0` 为 SR=1.0 下最优，但仍落后 baseline；`w=1.5` 虽更短但 SR 掉到 0.667（出现 timeout）。

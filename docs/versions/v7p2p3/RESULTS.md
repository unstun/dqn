# v7p2p3 结果（epsilon 修复验证）

## 数据来源
- 训练 run：`runs/v7p2p3_train300_esbest/train_20260221_003108`
- 推理 run：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529`
- 对照 run（v7p1 同口径）：`runs/v7p1_train300_esbest/train_20260221_001041/infer/20260221_002516`

## 一、训练完成情况
- `episodes_target=300`
- `episodes_completed=220`
- `stop_reason=rl_early_stop_plateau`
- `chosen_ckpt=best`

来源：`runs/v7p2p3_train300_esbest/train_20260221_003108/configs/train_meta_forest_a.json`

## 二、v7p2p3 推理指标（runs=5）
来源：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529/table2_kpis_mean_raw.csv`

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.600 | 19.0299 | 21.2000 | 0.199223 | 0.298 |
| short | Hybrid A*-MPC | 1.000 | 16.8724 | 10.0000 | 0.077256 | N/A |
| mid | CNN-DDQN | 0.200 | 24.1106 | 14.0500 | 0.061991 | 0.191 |
| mid | Hybrid A*-MPC | 1.000 | 25.1525 | 13.8700 | 0.079926 | N/A |
| long | CNN-DDQN | 0.600 | 54.3597 | 39.5500 | 0.192529 | 0.288 |
| long | Hybrid A*-MPC | 1.000 | 43.0247 | 22.8200 | 0.067176 | N/A |

## 三、与 v7p1_train300_esbest 对照（同 runs=5）
| 套件 | 指标 | v7p1 | v7p2p3 | delta(v7p2p3-v7p1) |
|---|---|---:|---:|---:|
| short | success_rate | 0.800 | 0.600 | -0.200 |
| short | avg_path_length | 16.3487 | 19.0299 | +2.6812 |
| short | path_time_s | 10.7750 | 21.2000 | +10.4250 |
| mid | success_rate | 0.800 | 0.200 | -0.600 |
| mid | avg_path_length | 30.4260 | 24.1106 | -6.3154 |
| mid | path_time_s | 19.2750 | 14.0500 | -5.2250 |
| long | success_rate | 0.800 | 0.600 | -0.200 |
| long | avg_path_length | 57.4691 | 54.3597 | -3.1094 |
| long | path_time_s | 37.4500 | 39.5500 | +2.1000 |

## 四、failure_reason 分布（CNN-DDQN, runs=5）
- v7p2p3：
  - short：`reached=3`, `timeout=2`
  - mid：`reached=1`, `collision=1`, `timeout=3`
  - long：`reached=3`, `timeout=2`
- v7p1_train300_esbest：
  - short：`reached=4`, `timeout=1`
  - mid：`reached=4`, `timeout=1`
  - long：`reached=4`, `timeout=1`

## 五、ε 衰减修复确认
- 调度函数：`linear_epsilon(episode, eps_start, eps_final, decay_episodes)`。
- 对比（`eps_start=0.2`, `eps_final=0.02`）：
  - `ep=220`：`v7p1=0.1912`, `v7p2p3=0.0477`
  - `ep=260`：`v7p1=0.1896`, `v7p2p3=0.0200`
- 结论：修复在机制上成立，但本轮任务口径下结果未优于 `v7p1`。

## 六、门槛检查（本次 runs=5，仅版本内决策）
- `success_rate(CNN) >= success_rate(Hybrid)`：short 未通过，mid 未通过，long 未通过。
- `avg_path_length(CNN) < avg_path_length(Hybrid)`：short 未通过，mid 通过，long 未通过。
- `path_time_s(CNN) < path_time_s(Hybrid)`：short 未通过，mid 未通过，long 未通过。
- 结论：`v7p2p3` 未通过。

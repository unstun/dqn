# v6p2p4 结果

## 数据来源
- 训练配置：`configs/v6p2p4.json`
- 训练 run：`runs/v6p2p4/train_20260219_153029`
- 推理 run：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252`
- KPI 文件：`runs/v6p2p4/train_20260219_153029/infer/20260219_161252/table2_kpis_mean_raw.csv`
- 运行记录：`docs/versions/v6p2p4/runs/README.md`

## 一、本轮执行结论
- 已完成 `cnn-ddqn / ddpg / sac` 三算法各 300 轮训练（`episodes_completed=300`，无提前终止）。
- 已完成 `short/mid/long` 三套件各 `runs=5` 推理，并与 `Hybrid A*-MPC` 对比。
- 在本轮设置下，`DDPG/SAC` 成功率均为 `0.00`，未形成可用对比基线；`CNN-DDQN` 仍显著优于 `DDPG/SAC`，但对 `Hybrid A*-MPC` 未通过门槛。

## 二、指标总表（runs=5）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | planning_time_s |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.80 | 18.1736 | 19.00 | 0.246483 | 0.59935 |
| short | DDPG | 0.00 | N/A | N/A | N/A | 0.04468 |
| short | SAC | 0.00 | N/A | N/A | N/A | 0.16730 |
| short | Hybrid A*-MPC | 1.00 | 16.8724 | 10.00 | 0.077256 | 0.82362 |
| mid | CNN-DDQN | 0.80 | 31.4601 | 19.45 | 0.180060 | 0.99823 |
| mid | DDPG | 0.00 | N/A | N/A | N/A | 0.08431 |
| mid | SAC | 0.00 | N/A | N/A | N/A | 0.13322 |
| mid | Hybrid A*-MPC | 1.00 | 25.1525 | 13.87 | 0.079926 | 0.31588 |
| long | CNN-DDQN | 1.00 | 51.9006 | 30.42 | 0.151765 | 0.66105 |
| long | DDPG | 0.00 | N/A | N/A | N/A | 0.00867 |
| long | SAC | 0.00 | N/A | N/A | N/A | 0.02647 |
| long | Hybrid A*-MPC | 1.00 | 43.0247 | 22.82 | 0.067176 | 3.93857 |

## 三、failure_reason 分布（runs=5）
- CNN-DDQN：
  - short：`reached=4`, `collision=1`
  - mid：`reached=4`, `timeout=1`
  - long：`reached=5`
- DDPG：
  - short：`collision=4`, `timeout=1`
  - mid：`collision=3`, `timeout=2`
  - long：`collision=5`
- SAC：
  - short：`collision=4`, `timeout=1`
  - mid：`collision=4`, `timeout=1`
  - long：`collision=5`
- Hybrid A*-MPC：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`

## 四、门槛检查（仅 runs=5，非最终结论）
- 对 `CNN-DDQN` vs `Hybrid A*-MPC`：
  - short：
    - `success_rate(CNN) >= success_rate(Hybrid)`：**否**（`0.80 < 1.00`）
    - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**否**（`18.1736 > 16.8724`）
    - `path_time_s(CNN) < path_time_s(Hybrid)`：**否**（`19.00 > 10.00`）
  - long：
    - `success_rate(CNN) >= success_rate(Hybrid)`：**是**（`1.00 = 1.00`）
    - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**否**（`51.9006 > 43.0247`）
    - `path_time_s(CNN) < path_time_s(Hybrid)`：**否**（`30.42 > 22.82`）
- 对 `DDPG/SAC` vs `Hybrid A*-MPC`：
  - short/long 两套件 `success_rate=0.00`，门槛均未满足。
- 结论：
  - 本轮 `runs=5` 下未通过门槛，且不构成最终结论；仍需 short/long 各 `runs=20` full 评测。

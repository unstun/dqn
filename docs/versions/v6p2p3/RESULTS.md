# v6p2p3 结果

## 数据来源
- 训练配置：`configs/v6p2p3.json`
- 训练 run（早停）：`runs/v6p2p3/train_20260219_135522`
- 训练 run（300 轮完成）：`runs/v6p2p3/train_20260219_142104`
- 推理 run：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315`
- KPI 文件：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315/table2_kpis_mean_raw.csv`
- 运行记录：`docs/versions/v6p2p3/runs/README.md`

## 一、本轮执行结论
- 已完成一次严格 300 轮训练（第二次训练，`episodes=300/300`）。
- 已完成 `short/mid/long` 三套件各 `runs=5` 推理。
- 本轮属于非最终门槛评测（最终门槛要求 short/long 各 `runs=20`）。

## 二、指标总表（runs=5）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | planning_time_s |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.80 | 15.7615 | 9.3625 | 0.064731 | 0.25962 |
| short | Hybrid A*-MPC | 1.00 | 16.8724 | 10.0000 | 0.077256 | 0.95450 |
| mid | CNN-DDQN | 1.00 | 25.3193 | 15.0600 | 0.109941 | 0.36453 |
| mid | Hybrid A*-MPC | 1.00 | 25.1525 | 13.8700 | 0.079926 | 0.36308 |
| long | CNN-DDQN | 1.00 | 46.2403 | 28.9500 | 0.203928 | 0.76663 |
| long | Hybrid A*-MPC | 1.00 | 43.0247 | 22.8200 | 0.067176 | 4.25005 |

## 三、short+long 平均（runs=5）
- CNN-DDQN：
  - `success_rate_mean = 0.9000`
  - `avg_path_length_mean = 31.0009`
  - `path_time_s_mean = 19.1563`
- Hybrid A*-MPC：
  - `success_rate_mean = 1.0000`
  - `avg_path_length_mean = 29.9486`
  - `path_time_s_mean = 16.4100`

## 四、failure_reason 分布（runs=5）
- CNN-DDQN：
  - short：`reached=4`, `collision=1`
  - mid：`reached=5`
  - long：`reached=5`
  - 汇总：`reached=14`, `collision=1`
- Hybrid A*-MPC：
  - short：`reached=5`
  - mid：`reached=5`
  - long：`reached=5`
  - 汇总：`reached=15`

## 五、门槛检查（仅基于 runs=5，非最终结论）
- short：
  - `success_rate(CNN) >= success_rate(Hybrid)`：**否**（`0.80 < 1.00`）
  - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**是**（`15.7615 < 16.8724`）
  - `path_time_s(CNN) < path_time_s(Hybrid)`：**是**（`9.3625 < 10.0000`）
- long：
  - `success_rate(CNN) >= success_rate(Hybrid)`：**是**（`1.00 = 1.00`）
  - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**否**（`46.2403 > 43.0247`）
  - `path_time_s(CNN) < path_time_s(Hybrid)`：**否**（`28.9500 > 22.8200`）
- 结论：
  - 当前 `runs=5` 下未满足对标条件，且不构成最终门槛结论。
  - 仍需 short/long 各 `runs=20` 的 full 评测。

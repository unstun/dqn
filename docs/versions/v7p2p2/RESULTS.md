# v7p2p2 结果（smoke 失败归档）

## 数据来源
- 训练 run：`runs/v7p2p2_smoke150/train_20260220_230753`
- 推理 run（v7p2p2，runs=3）：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053`
- 对照推理 run（v7p1，runs=3）：`runs/v7p1_remote150_eval3/20260220_232121`
- KPI：
  - `runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053/table2_kpis_mean_raw.csv`
  - `runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053/table2_kpis_raw.csv`
  - `runs/v7p1_remote150_eval3/20260220_232121/table2_kpis_mean_raw.csv`
  - `runs/v7p1_remote150_eval3/20260220_232121/table2_kpis_raw.csv`

## 一、v7p2p2 对基线（Hybrid A*-MPC）结果（runs=3）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 16.1713 | 12.3750 | 0.152800 | 0.139 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A |
| mid | CNN-DDQN | 0.667 | 29.4152 | 23.5000 | 0.230717 | 0.418 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A |
| long | CNN-DDQN | 0.333 | 66.6176 | 44.2500 | 0.190042 | 0.485 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A |

## 二、与 v7p1（150轮模型）同口径对照（runs=3）
| 套件 | 指标 | v7p1_remote150 | v7p2p2_smoke150 | delta(v7p2p2-v7p1) |
|---|---|---:|---:|---:|
| short | success_rate | 1.000 | 0.667 | -0.333 |
| short | avg_path_length | 16.7909 | 16.1713 | -0.6196 |
| short | path_time_s | 15.6667 | 12.3750 | -3.2917 |
| short | avg_curvature_1_m | 0.121277 | 0.152800 | +0.031523 |
| mid | success_rate | 0.667 | 0.667 | +0.000 |
| mid | avg_path_length | 26.8669 | 29.4152 | +2.5483 |
| mid | path_time_s | 23.5750 | 23.5000 | -0.0750 |
| mid | avg_curvature_1_m | 0.146422 | 0.230717 | +0.084295 |
| long | success_rate | 0.333 | 0.333 | +0.000 |
| long | avg_path_length | 43.5187 | 66.6176 | +23.0989 |
| long | path_time_s | 28.1500 | 44.2500 | +16.1000 |
| long | avg_curvature_1_m | 0.046060 | 0.190042 | +0.143982 |

## 三、failure_reason 分布（CNN-DDQN，runs=3）
- v7p1:
  - short：`reached=3`
  - mid：`reached=2`, `collision=1`
  - long：`reached=1`, `timeout=2`
- v7p2p2:
  - short：`reached=2`, `collision=1`
  - mid：`reached=2`, `timeout=1`
  - long：`reached=1`, `timeout=2`

## 四、ε 衰减修复确认
- 调度函数：`linear_epsilon(episode, eps_start, eps_final, decay_episodes)`。
- 在 `eps_start=0.2`, `eps_final=0.02` 下：
  - `v7p1 (eps_decay=4500)`：`episode=150 -> epsilon≈0.1940`
  - `v7p2p2 (eps_decay=200)`：`episode=150 -> epsilon≈0.0650`
- 结论：修复生效（探索率显著下降），但本轮 smoke 指标未体现稳定收益。

## 五、门槛检查（仅 smoke 决策，不作为最终科研结论）
- `success_rate(CNN) >= success_rate(Hybrid)`：short 未通过，mid 未通过，long 未通过。
- `avg_path_length(CNN) < avg_path_length(Hybrid)`：short 通过，mid 未通过，long 未通过。
- `path_time_s(CNN) < path_time_s(Hybrid)`：short 未通过，mid 未通过，long 未通过。
- 决策：**未通过 smoke 门，不进入 full。**

# v6p3 结果

## 数据来源
- 主配置：`configs/v6p3.json`
- 复现配置：`configs/repro_20260219_v6p3_cnn_ddqn_vs_hybrid_astar_mpc.json`
- 训练 run：`runs/v6p3_smoke120/train_20260219_041557`
- 推理 run：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629`
- KPI 文件：
  - 均值：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629/table2_kpis_mean_raw.csv`
  - 逐回合：`runs/v6p3_smoke120/train_20260219_041557/infer/20260219_043629/table2_kpis_raw.csv`

## 一、本轮执行结论
- 已完成 smoke：`120` 轮训练 + `short/mid/long` 各 `runs=3` 推理。
- `CNN-DDQN`：`short=1.000`，`mid=0.333`，`long=1.000`。
- `Hybrid A*-MPC`：`short/mid/long` 均为 `1.000`。
- `mid` 套件中 `CNN-DDQN` 失败类型为 `collision`（2/3），是当前主要回退点。
- `v6p3` 的唯一 RL 算法为 `CNN-DDQN`，唯一对比基线为 `Hybrid A*-MPC`。

## 二、指标总表（smoke: short/mid/long, runs=3）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | planning_time_s | argmax_inadmissible_rate | failure_reason |
|---|---|---:|---:|---:|---:|---:|---|
| short | CNN-DDQN | 1.000 | 17.4834 | 12.5167 | 0.58282 | 0.208 | reached=3 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 1.46233 | N/A | reached=3 |
| mid | CNN-DDQN | 0.333 | 24.5929 | 16.5500 | 1.27164 | 0.346 | reached=1, collision=2 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.39626 | N/A | reached=3 |
| long | CNN-DDQN | 1.000 | 50.1763 | 34.2333 | 1.19457 | 0.256 | reached=3 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 3.92923 | N/A | reached=3 |

## 三、failure_reason 汇总
- `CNN-DDQN`：`reached=7, collision=2`
- `Hybrid A*-MPC`：`reached=9`

## 四、门槛检查（short/long + runs=20）
- `N/A`（本轮为 smoke，非最终 `runs=20` 口径）

## 五、待补动作
1. 运行 full：`300` 轮训练 + `short/long` 至少 `runs=20`。
2. 对 `mid` 失败样本做 trace 复盘，确认 `collision` 触发机制与动作模式。

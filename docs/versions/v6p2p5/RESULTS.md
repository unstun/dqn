# v6p2p5 结果

## 数据来源
- 主配置：`configs/v6p2p5.json`
- 复现配置：`configs/repro_20260219_v6p2p5_cont_isolation.json`
- 训练 run：`runs/v6p2p5_smoke120/train_20260219_030737`
- 推理 run：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302`
- KPI 文件：
  - 均值：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302/table2_kpis_mean_raw.csv`
  - 逐回合：`runs/v6p2p5_smoke120/train_20260219_030737/infer/20260219_033302/table2_kpis_raw.csv`

## 一、本轮执行结论（smoke）
- 已完成 smoke：训练 `120` 轮 + `short/mid/long` 各 `runs=3` 推理。
- `CNN-DDQN` 在 `short` 与 `long` 达到 `SR=1.00`，`mid` 为 `SR=0.667`。
- `DDPG` 仅在 `short` 成功 `1/3`，`mid/long` 为 `0/3`。
- `SAC` 在 `short/mid/long` 均为 `SR=0.00`。
- 连续算法在 smoke 下仍以 `timeout` 为主失败模式，且 `DDPG` 的 `argmax_inadmissible_rate` 持续偏高。

## 二、指标总表（smoke: short/mid/long, runs=3）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | planning_time_s | argmax_inadmissible_rate | failure_reason |
|---|---|---:|---:|---:|---:|---:|---|
| short | CNN-DDQN | 1.000 | 16.3946 | 10.1333 | 0.32395 | 0.107 | reached=3 |
| short | DDPG | 0.333 | 11.9818 | 35.0500 | 4.63001 | 0.553 | reached=1, timeout=2 |
| short | SAC | 0.000 | N/A | N/A | 3.12176 | 0.014 | timeout=3 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 1.37740 | N/A | reached=3 |
| mid | CNN-DDQN | 0.667 | 26.0232 | 17.3500 | 1.30333 | 0.400 | reached=2, timeout=1 |
| mid | DDPG | 0.000 | N/A | N/A | 5.99267 | 0.766 | timeout=3 |
| mid | SAC | 0.000 | N/A | N/A | 4.00964 | 0.313 | timeout=3 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.36300 | N/A | reached=3 |
| long | CNN-DDQN | 1.000 | 52.2547 | 30.0833 | 0.70216 | 0.128 | reached=3 |
| long | DDPG | 0.000 | N/A | N/A | 7.41043 | 0.943 | timeout=3 |
| long | SAC | 0.000 | N/A | N/A | 3.02035 | 0.000 | timeout=3 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 3.96724 | N/A | reached=3 |

## 三、failure_reason 汇总（3 套件合并）
- CNN-DDQN：`reached=8, timeout=1`
- DDPG：`reached=1, timeout=8`
- SAC：`timeout=9`
- Hybrid A*-MPC：`reached=9`

## 四、门槛检查（short/long + runs=20）
- `N/A`（本轮为 smoke，非最终 `runs=20` 口径）。

## 五、下一步建议
1. 保持 `v6p2p5` 隔离结构不动，先对连续算法做单独 smoke 调参（仅改 `cont-*` 参数）。
2. 若连续算法 smoke 仍 `SR=0`，增加连续专用奖励/动作约束诊断并记录到同一版本留档。
3. 满足 smoke 改善后再进入 full（`runs=20`）评测。

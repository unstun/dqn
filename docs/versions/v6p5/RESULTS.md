# v6p5 结果

## 数据来源
- 主配置：`configs/v6p5.json`
- 复现配置：`configs/repro_20260220_v6p5_local_obs_mid360_nearfull.json`
- 复现配置（BN）：`configs/repro_20260220_v6p5_cnn_batchnorm.json`
- 复现配置（CUDA 优先）：`configs/repro_20260220_v6p5_cuda_pref_gpu_throughput.json`
- 本轮有效训练 run：`runs/v6p5_bn_quick20/train_20260219_165821`
- 本轮有效推理 run：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130`
- KPI 文件：
  - `runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130/table2_kpis_mean_raw.csv`
  - `runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130/table2_kpis_raw.csv`

## 一、本轮执行结论
- 已完成 `v6p5` 的局部高分辨率观测改造与 `CNNQNetwork`（CNN 版 Q 网络）`BatchNorm2d` 改造。
- `2026-02-19` 完成一次可用快速试跑（`quick20`，`episodes=20` + `runs=3`），用于先验检查 BN 版本链路可运行。
- 本轮尚未完成标准 smoke（`episodes=300`）与 full（`runs=20`），因此结果仅作趋势参考，不作为最终结论。
- `2026-02-19` 标准 smoke 已尝试：`runs/v6p5_bn_smoke300/train_20260219_170903`（`--device cuda`），因人工中断未形成可用 KPI（记为 `N/A`）。

## 二、指标总表（short/mid/long）
> 口径：`quick20`（非标准 smoke），`runs=3`。

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | planning_time_s | argmax_inadmissible_rate | failure_reason |
|---|---|---:|---:|---:|---:|---:|---|
| short | CNN-DDQN | 1.000 | 24.6382 | 29.0500 | 3.3390 | 0.420 | reached=3 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 1.2341 | N/A | reached=3 |
| mid | CNN-DDQN | 0.333 | 50.8339 | 55.3000 | 3.9756 | 0.445 | reached=1, timeout=2 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.3475 | N/A | reached=3 |
| long | CNN-DDQN | 0.000 | N/A | N/A | 3.3480 | 0.601 | timeout=2, collision=1 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 3.4793 | N/A | reached=3 |

## 三、failure_reason 汇总
- `CNN-DDQN`（9 回合）：`reached=4`，`timeout=4`，`collision=1`
- `Hybrid A*-MPC`（9 回合）：`reached=9`

## 四、门槛检查（short/long + runs=20）
- `N/A`（本轮仅 `quick20` 先验试跑，未执行标准 smoke/full 与 `runs=20`）

## 五、待补动作
1. 按标准 smoke 口径执行：`episodes=300, runs=3`（不带 quick override）。
2. smoke 通过后执行 full：`runs=20`，并做 short/long 门槛判定。
3. 将 full 的 `failure_reason` 分布与门槛结论回填到版本索引。

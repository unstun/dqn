# v7p1 结果对比

## 数据来源
- KPI（均值）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/table2_kpis_raw.csv`
- 运行口径：`runs=5`（来自 `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927/configs/run.json`）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 1.00 | 16.7849 | 11.60 | 0.174422 |
| short | Hybrid A*-MPC | 1.00 | 16.8724 | 10.00 | 0.077256 |
| mid | CNN-DDQN | 1.00 | 26.6106 | 16.60 | 0.219962 |
| mid | Hybrid A*-MPC | 1.00 | 25.1525 | 13.87 | 0.079926 |
| long | CNN-DDQN | 1.00 | 51.9081 | 34.01 | 0.214457 |
| long | Hybrid A*-MPC | 1.00 | 43.0247 | 22.82 | 0.067176 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 通过 | 通过 | 不通过 | 未通过 |
| mid | 通过 | 不通过 | 不通过 | 未通过 |
| long | 通过 | 不通过 | 不通过 | 未通过 |

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本次仅 runs=5）
- long（runs=20）：`N/A`（本次仅 runs=5）
- 结论：当前证据不可用于最终门槛结论。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：short=`reached=5`，mid=`reached=5`，long=`reached=5`（总计 `reached=15`）
- Hybrid A*-MPC：short=`reached=5`，mid=`reached=5`，long=`reached=5`（总计 `reached=15`）

## 附：v8 推理期消融（基于 v7p1 checkpoint，infer-only，runs=3）

> 目的：在保持 `success_rate≈1.0` 前提下，先验证推理期 admissible gating（可采纳动作判定）的参数敏感性，判断“只调推理期规则”对 `avg_path_length/path_time_s` 的上限。

### 数据来源（均值 KPI）
- R0：`runs/v8_ablate_v7p1_R0_h30_mp001/20260223_004917/table2_kpis_mean_raw.csv`
- R1：`runs/v8_ablate_v7p1_R1_h30_mp000/20260223_005010/table2_kpis_mean_raw.csv`
- R2：`runs/v8_ablate_v7p1_R2_h15_mp001/20260223_005037/table2_kpis_mean_raw.csv`
- R3：`runs/v8_ablate_v7p1_R3_h15_mp000/20260223_005102/table2_kpis_mean_raw.csv`
- R4：`runs/v8_ablate_v7p1_R4_h15_mpn002/20260223_005128/table2_kpis_mean_raw.csv`
- R5：`runs/v8_ablate_v7p1_R5_strict_argmax/20260223_005153/table2_kpis_mean_raw.csv`

说明：
- 固定同一模型：`runs/v7p1_train300_esbest/train_20260221_010743`
- 固定对照 baseline：`Hybrid A*-MPC`
- 本轮 R0–R4 的 baseline（Hybrid A*-MPC）均值 KPI 行逐项一致（同一组随机 pairs，便于横向对比 RL 指标）。

### 结果汇总（CNN-DDQN；runs=3 均值）

| run | `adm_h` | `min_progress_m` | short SR | short L | short T | mid SR | mid L | mid T | long SR | long L | long T | long inad | long fb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 30 | 0.01 | 1.000 | 16.2020 | 10.3333 | 1.000 | 24.9075 | 15.9500 | 1.000 | 54.7787 | 37.4000 | 0.303 | 0.303 |
| R1 | 30 | 0.00 | 1.000 | 16.2020 | 10.3333 | 1.000 | 24.9075 | 15.9500 | 1.000 | 54.4759 | 33.3333 | 0.249 | 0.244 |
| R2 | 15 | 0.01 | 1.000 | 17.3391 | 11.2500 | 0.667 | 25.0307 | 17.1000 | 1.000 | 48.8966 | 28.9333 | 0.091 | 0.091 |
| R3 | 15 | 0.00 | 1.000 | 17.3391 | 11.2500 | 0.667 | 25.0307 | 17.1000 | 1.000 | 48.8966 | 28.9333 | 0.091 | 0.091 |
| R4 | 15 | -0.02 | 1.000 | 17.3282 | 11.2500 | 1.000 | 24.6942 | 20.8167 | 0.667 | 47.8390 | 27.6500 | 0.241 | 0.241 |
| R5 | 30 | 0.01 | 0.000 | N/A | N/A | 0.000 | N/A | N/A | 0.000 | N/A | N/A | 0.663 | 0.000 |

注：
- `long inad` = `argmax_inadmissible_rate`（argmax 选到不可采纳动作比例）
- `long fb` = `fallback_rate`（最终动作 != 原始 argmax(Q) 的比例）

### baseline（Hybrid A*-MPC；本轮各组一致）

| 套件 | success_rate | avg_path_length | path_time_s |
|---|---:|---:|---:|
| short | 1.000 | 17.0342 | 10.2667 |
| mid | 1.000 | 24.0814 | 13.3333 |
| long | 1.000 | 43.0107 | 22.8167 |

### 结论（用于 v8 决策）
- R1（`min_progress_m=0.0`）相对 R0（`min_progress_m=0.01`）：long `path_time_s 37.4000 -> 33.3333`（约 **-10.9%**），但 `avg_path_length 54.7787 -> 54.4759` 基本不变。
- `adm_h=15`（R2/R3/R4）虽然能进一步压 long 的 `path_time_s/avg_path_length`，但会引入 mid/long `timeout` 导致 `success_rate<1.0`，不满足“SR≈1.0 前提下压 path/time”的约束。
- strict-argmax（R5，`--forest-no-fallback`）short/mid/long 全 `collision`（`success_rate=0`），说明当前 checkpoint 强依赖 `shielded/hybrid` 推理兜底；仅靠关闭兜底不可行。
- 推理期 gating 微调对 `path_time_s` 有一定空间，但对 `avg_path_length` 的改善有限；v8 若要系统性压路径/时间，优先考虑训练侧改动（进度定义/奖励/专家约束对齐等）。

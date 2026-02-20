# v7p2 结果

## 数据来源
- 训练配置：`configs/v7p2.json`
- 复现配置：`configs/repro_20260220_v7p2_markov_obs_prev_a.json`
- 训练 run：`runs/v7p2_smoke/train_20260220_211732`
- 推理 run：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137`
- KPI 文件：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137/table2_kpis_mean_raw.csv`
- 明细文件：`runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137/table2_kpis_raw.csv`
- 运行记录：`docs/versions/v7p2/runs/README.md`

## 一、本轮执行结论
- 本版已完成：`self-check -> smoke`。
- smoke 口径：训练 `episodes=40`，推理 `short/mid/long` 各 `runs=3`。
- 本版目标是观测一致性修复，不是最终门槛结论版。

## 二、指标总表（runs=3）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | planning_time_s |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 20.0492 | 11.6833 | 0.134386 | 0.14860 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | 0.50737 |
| mid | CNN-DDQN | 0.333 | 26.9989 | 14.8000 | 0.150952 | 0.24603 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | 0.16280 |
| long | CNN-DDQN | 1.000 | 65.6659 | 44.5333 | 0.218101 | 0.45608 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | 1.41956 |

## 三、short+long 平均（runs=3）
- CNN-DDQN：
  - `success_rate_mean = 1.0000`
  - `avg_path_length_mean = 42.8576`
  - `path_time_s_mean = 28.1083`
- Hybrid A*-MPC：
  - `success_rate_mean = 1.0000`
  - `avg_path_length_mean = 30.0225`
  - `path_time_s_mean = 16.5417`

## 四、failure_reason 分布（runs=3）
- CNN-DDQN：
  - short：`reached=3`
  - mid：`collision=1`, `reached=1`, `timeout=1`
  - long：`reached=3`
  - 汇总：`reached=7`, `collision=1`, `timeout=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 汇总：`reached=9`

## 五、门槛检查（基于 runs=3，非最终结论）
- short：
  - `success_rate(CNN) >= success_rate(Hybrid)`：**是**（`1.00 = 1.00`）
  - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**否**（`20.0492 > 17.0342`）
  - `path_time_s(CNN) < path_time_s(Hybrid)`：**否**（`11.6833 > 10.2667`）
- long：
  - `success_rate(CNN) >= success_rate(Hybrid)`：**是**（`1.00 = 1.00`）
  - `avg_path_length(CNN) < avg_path_length(Hybrid)`：**否**（`65.6659 > 43.0107`）
  - `path_time_s(CNN) < path_time_s(Hybrid)`：**否**（`44.5333 > 22.8167`）
- 结论：
  - 当前 smoke 数据未满足最终门槛；且本轮不是 full 评测。
  - 仍需 short/long 各 `runs=20` 的 full 评测后再做最终结论。

## 六、追加 full300（best vs final，对比 runs=20）

### 6.1 数据来源
- best 训练 run：`runs/v7p2_full300/train_20260220_213003`
- best 推理 run：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341`
- best KPI（均值）：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341/table2_kpis_mean_raw.csv`
- best KPI（逐回合）：`runs/v7p2_full300/train_20260220_213003/infer/20260220_214341/table2_kpis_raw.csv`
- final 训练 run：`runs/v7p2_final300/train_20260220_215145`
- final 推理 run：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346`
- final KPI（均值）：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346/table2_kpis_mean_raw.csv`
- final KPI（逐回合）：`runs/v7p2_final300/train_20260220_215145/infer/20260220_220346/table2_kpis_raw.csv`

### 6.2 CNN-DDQN 对比（best -> final）
| 套件 | success_rate | avg_path_length | path_time_s | planning_time_s | argmax_inadmissible_rate |
|---|---:|---:|---:|---:|---:|
| short | `0.85 -> 0.80` | `20.2527 -> 19.8434` | `13.2912 -> 12.7531` | `0.18406 -> 0.26475` | `0.187 -> 0.209` |
| mid | `0.65 -> 0.85` | `33.1484 -> 28.0888` | `20.4538 -> 17.2441` | `0.34007 -> 0.25515` | `0.238 -> 0.207` |
| long | `0.75 -> 0.75` | `62.5007 -> 66.5743` | `39.0567 -> 42.6033` | `0.55093 -> 2.8190` | `0.324 -> 0.345` |

### 6.3 failure_reason 分布（CNN-DDQN）
- best：
  - short：`reached=17`, `timeout=1`, `collision=2`
  - mid：`reached=13`, `collision=4`, `timeout=3`
  - long：`reached=15`, `timeout=5`
- final：
  - short：`reached=16`, `collision=4`
  - mid：`reached=17`, `collision=3`
  - long：`reached=15`, `timeout=4`, `collision=1`

### 6.4 对比结论
- `final` 在 mid 套件明显提升（成功率、路径长度、时间均改善）。
- `final` 在 short 套件出现成功率下降（`0.85 -> 0.80`），但成功样本路径更短、耗时更低。
- `final` 在 long 套件没有成功率收益，且路径与耗时劣化，`planning_time_s` 明显增大。
- 结论：`final` 不能替代 `best` 作为 `v7p2` 的统一最优 checkpoint（检查点）；更适合作为“mid 优化、long 风险增大”的对照实验记录。

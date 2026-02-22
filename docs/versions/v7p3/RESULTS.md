# v7p3 结果对比

## 数据来源
- KPI（均值）：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 单测
- 命令：`conda run -n ros2py310 python -m pytest tests/test_v7p3_adaptive_penalty.py -v`
- 结果：通过（`3 passed`）

### 2) 最小自检
- 本地：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 远端（`ubuntu-zt`）：
  - `ssh ubuntu-zt "... python train.py --self-check"`
  - `ssh ubuntu-zt "... python infer.py --self-check"`
- 结果：通过（CUDA 可用）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 25.3610 | 17.4500 | 0.193361 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 34.0156 | 18.3250 | 0.148345 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 57.1796 | 30.1500 | 0.102428 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p2p10` 趋势（仅 smoke 门参考）
- short：`SR` 持平（`0.667 -> 0.667`），`path/time/smoothness` 改善（`26.7961/25.05/0.269 -> 25.361/17.45/0.193`）。
- mid：`SR` 提升（`0.333 -> 0.667`），但 `path/time/smoothness` 退化（`24.0931/13.35/0.0797 -> 34.0156/18.325/0.1483`）。
- long：`SR` 持平（`0.333 -> 0.333`），`path/time` 退化（`51.0399/27.35 -> 57.1796/30.15`）。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）
- long（runs=20）：`N/A`（本轮仅 smoke）
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=2`，`collision=1`
  - long：`reached=1`，`timeout=1`，`collision=1`
  - 合计：`reached=5`，`timeout=2`，`collision=2`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：虽然 short 与 mid 局部有改善，但 long `path/time` 退化且三套件对基线三项门槛仍全部不通过，未形成“全套件明确收益”。
- 处理：`v7p3` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。

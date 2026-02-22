# v7p3p2 结果对比

## 数据来源
- KPI（均值）：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 单测
- 命令：`conda run -n ros2py310 python -m pytest tests/test_v7p3_adaptive_penalty.py tests/test_v7p3p1_adaptive_penalty_generalized.py tests/test_v7p3p2_turn_aware_topk.py -q`
- 结果：通过（`9 passed`）

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
| short | CNN-DDQN | 0.333 | 27.4510 | 19.2000 | 0.332357 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 33.7576 | 22.7500 | 0.246386 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 55.8795 | 33.8000 | 0.165122 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 相对 `v7p3p1` 趋势（仅 smoke 门参考）
- short：`SR` 下降（`0.667 -> 0.333`），`path/time` 回落（`32.0322/24.375 -> 27.451/19.2`），但曲率上升（`0.3025 -> 0.3324`）。
- mid：`SR` 下降（`1.000 -> 0.667`），`path/time` 小幅回落（`34.4515/22.7667 -> 33.7576/22.75`），曲率上升（`0.1896 -> 0.2464`）。
- long：`SR` 下降（`1.000 -> 0.333`），`path/time/curvature` 同步回落（`83.8999/48.1/0.2185 -> 55.8795/33.8/0.1651`）。

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）
- long（runs=20）：`N/A`（本轮仅 smoke）
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=1`，`timeout=2`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=1`，`timeout=2`
  - 合计：`reached=4`，`timeout=5`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：尽管 long 的 `path/time` 显著回落，但 `success_rate` 在三套件明显下降，且对基线三项门槛全部未通过。
- 处理：`v7p3p2` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。

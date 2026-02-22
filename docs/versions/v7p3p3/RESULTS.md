# v7p3p3 结果对比

## 数据来源
- KPI（均值）：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 单测
- 命令（期望）：`conda run -n ros2py310 python -m pytest tests/test_v7p3_adaptive_penalty.py tests/test_v7p3p1_adaptive_penalty_generalized.py tests/test_v7p3p2_turn_aware_topk.py -q`
- 结果：**未执行**（当前 `ros2py310` 环境缺少 `pytest`：`No module named pytest`）。

### 2) 最小自检
- 远端（`ubuntu-zt`）：
  - `ssh ubuntu-zt "... python train.py --self-check"`
  - `ssh ubuntu-zt "... python infer.py --self-check"`
- 结果：通过（CUDA 可用）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

> 说明：当 `success_rate=0` 时，`avg_path_length/path_time_s/avg_curvature_1_m` 为 `N/A`（该套件无到达样本）。

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.000 | N/A | N/A | N/A |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 26.4057 | 14.6000 | 0.128175 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 61.6431 | 32.4000 | 0.151448 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 相对 `v7p3p2` 趋势（仅 smoke 门参考）
- short：`SR` 下降（`0.333 -> 0.000`），出现 `collision/timeout` 集中，无法统计 `path/time`。
- mid：`SR` 下降（`0.667 -> 0.333`），但 `path/time/curvature` 明显回落（`33.7576/22.75/0.2464 -> 26.4057/14.6/0.1282`）。
- long：`SR` 上升（`0.333 -> 0.667`），`time/curvature` 小幅回落（`33.8/0.1651 -> 32.4/0.1514`），但 `path` 回升（`55.8795 -> 61.6431`）。

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | N/A | N/A | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）
- long（runs=20）：`N/A`（本轮仅 smoke）
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`collision=1`，`timeout=2`
  - mid：`reached=1`，`collision=1`，`timeout=1`
  - long：`reached=2`，`timeout=1`
  - 合计：`reached=3`，`timeout=4`，`collision=2`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：尽管 long 的 `success_rate` 从 `0.333` 回升到 `0.667`，但 short/mid 的 `success_rate` 明显下降，且 short 出现 `collision` 并完全未到达。
- 处理：`v7p3p3` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。


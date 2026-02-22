# v7p3p6 结果对比

## 数据来源
- 训练 run：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007`
- 推理 KPI（均值）：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831/table2_kpis_mean_raw.csv`
- 推理 KPI（逐回合）：`runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831/table2_kpis_raw.csv`

## 代码级验证结果

### 1) 最小自检（远端 `ubuntu-zt`）
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：通过（CUDA 可用）

### 2) smoke 流程
- `self-check -> smoke(train episodes=150) -> smoke infer(runs=3)`
- 状态：已完成

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 24.1966 | 13.5000 | 0.183115 | 0.319 | 0.319 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 0.333 | 29.0903 | 16.9500 | 0.221496 | 0.123 | 0.123 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 0.333 | 67.9985 | 39.5500 | 0.219173 | 0.380 | 0.380 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 否（0.667 < 1.000） | 否（24.1966 > 17.0342） | 否（13.5000 > 10.2667） | 不通过 |
| long | 否（0.333 < 1.000） | 否（67.9985 > 43.0107） | 否（39.5500 > 22.8167） | 不通过 |

## `failure_reason` 分布
- CNN-DDQN（总体）：`reached=4`，`timeout=5`
- Hybrid A*-MPC（总体）：`reached=9`
- CNN-DDQN（分套件）：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=1`，`timeout=2`
  - long：`reached=1`，`timeout=2`

## 相对上一版（v7p3p5）变化
- short：`SR` 持平（`0.667`），`path_time_s` 明显下降（`28.2 -> 13.5`），`avg_path_length` 略升（`23.0936 -> 24.1966`）。
- mid：`SR` 持平（`0.333`），`avg_path_length/path_time_s` 显著下降（`69.6498/39.8500 -> 29.0903/16.9500`）。
- long：`SR` 从 `0.000` 回升到 `0.333`，不再 `3/3 timeout`，但路径与时间仍远高于 baseline。

## smoke 门结论（go/no-go）
- 结论：`NO-GO（失败归档）`
- 原因：
  - short 与 long 均未满足三条门槛不等式；
  - long 仍有 `2/3 timeout`，且 `avg_path_length/path_time_s` 明显落后 baseline；
  - 但相较 `v7p3p5`，long 可达性已有恢复，说明当前调参方向有效。

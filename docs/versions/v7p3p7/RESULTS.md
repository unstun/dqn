# v7p3p7 结果对比

## 数据来源
- 训练 run：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248`
- 推理 KPI（均值）：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_mean_raw.csv`
- 推理 KPI（逐回合）：`runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329/table2_kpis_raw.csv`

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
| short | CNN-DDQN | 1.000 | 21.3138 | 13.9000 | 0.221307 | 0.130 | 0.130 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 1.000 | 36.6758 | 21.2500 | 0.204348 | 0.325 | 0.325 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 0.333 | 66.8440 | 35.7500 | 0.179354 | 0.240 | 0.240 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 是（1.000 = 1.000） | 否（21.3138 > 17.0342） | 否（13.9000 > 10.2667） | 不通过 |
| long | 否（0.333 < 1.000） | 否（66.8440 > 43.0107） | 否（35.7500 > 22.8167） | 不通过 |

## `failure_reason` 分布
- CNN-DDQN（总体）：`reached=7`，`timeout=2`
- Hybrid A*-MPC（总体）：`reached=9`
- CNN-DDQN（分套件）：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=1`，`timeout=2`

## 相对上一版（v7p3p6）变化
- short：`SR 0.667 -> 1.000`，`avg_path_length 24.1966 -> 21.3138`，`path_time_s 13.5 -> 13.9`（略升）。
- mid：`SR 0.333 -> 1.000`，但 `avg_path_length/path_time_s` 从 `29.0903/16.9500` 退化到 `36.6758/21.2500`。
- long：`SR` 持平 `0.333`，`avg_path_length/path_time_s` 从 `67.9985/39.5500` 改善到 `66.8440/35.7500`。
- 超时：CNN 总体从 `timeout=5` 降到 `timeout=2`，但 long 套件仍 `2/3 timeout`。

## smoke 门结论（go/no-go）
- 结论：`NO-GO（失败归档）`
- 原因：
  - short 与 long 仍未同时满足三条门槛不等式；
  - long 可达性无新增突破（`SR=0.333`、`2/3 timeout`）；
  - 虽然总体 timeout 从 5 降到 2，且 short/mid SR 升至 1.0，但不足以支撑进入 full 评测。

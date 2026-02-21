# v7p2p10 结果对比

## 数据来源
- KPI（均值）：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 远端 self-check
- 命令：
  - `python train.py --self-check`
  - `python infer.py --self-check`
- 环境：`ubuntu-zt` + `conda run -n ros2py310`
- 结果：通过（CUDA 可用，`device_ok=cuda:0`）。

### 2) 训练日志机制验收（关键摘录）
- `train_flow.log` 显示：
  - `Dynamic two-suite curriculum enabled`
  - `short_prob` 动态更新（约 `0.332~0.630` 区间）
  - 单变量为 `forest_reward_no_progress_penalty=0.35`
  - `grad_clip_hit_rate` 后段约 `0.997`，训练流程可完成

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 26.7961 | 25.0500 | 0.269396 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 24.0931 | 13.3500 | 0.079707 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 51.0399 | 27.3500 | 0.164422 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p2p9` 的趋势对照（仅作 smoke 门参考）
- `v7p2p9` CNN：short/mid/long `SR=0.667/0.333/0.000`。
- `v7p2p10` CNN：short/mid/long `SR=0.667/0.333/0.333`。
- 结论：long 有恢复、mid 指标改善，但 short path/smoothness 明显退化，整体仍不满足 go 条件。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）。
- long（runs=20）：`N/A`（本轮仅 smoke）。
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=1`，`timeout=2`
  - long：`reached=1`，`timeout=1`，`collision=1`
  - 合计：`reached=4`，`timeout=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：虽然 long 从 `0.0` 回升到 `0.333`，但 short 的 path/smoothness 明显劣化，且三套件对基线三项门槛仍全部不通过。
- 处理：`v7p2p10` 失败归档，保持当前代码继续前向迭代，不执行代码回退。

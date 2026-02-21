# v7p2p7 结果对比

## 数据来源
- KPI（均值）：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 本地单测
- 命令：
  - `conda run -n ros2py310 python -m pytest tests/test_agent_training_controls.py tests/test_globalcnn_network.py -v`
- 结果：`8 passed`

### 2) self-check
- 本地：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 远端（ubuntu-zt）：
  - `python train.py --self-check`
  - `python infer.py --self-check`
- 结果：均通过（CUDA 可用）。

## 训练日志机制验收（关键摘录）
- `train_flow.log` 显示：
  - `reward_scale=0.1000, reward_clip_abs=10.0000, grad_clip_norm=10.0000`
  - `Demo pretrain done: steps=14000/30000`
  - `RL progress ... grad_norm_ema=..., grad_clip_hit_rate=0.999~1.000`

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 24.6699 | 14.0000 | 0.267312 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 36.4765 | 24.5500 | 0.234826 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 61.7273 | 32.2000 | 0.154668 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p2p6` 的趋势对照（仅作 smoke 门参考）
- `v7p2p6` CNN：short/mid/long `SR=1.000/0.667/0.000`。
- `v7p2p7` CNN：short/mid/long `SR=0.333/0.667/0.333`。
- 结论：long 有恢复（`0.000 -> 0.333`），但 short 明显退化（`1.000 -> 0.333`），整体仍不满足 go 条件。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）。
- long（runs=20）：`N/A`（本轮仅 smoke）。
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`timeout=2`，`reached=1`
  - mid：`reached=2`，`timeout=1`
  - long：`timeout=1`，`collision=1`，`reached=1`
  - 合计：`timeout=4`，`reached=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：long 虽恢复但 short 显著退化，且三套件 path/time 仍全面落后基线。
- 处理：`v7p2p7` 失败归档，保持当前代码继续前向迭代，不执行代码回退。

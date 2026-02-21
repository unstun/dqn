# v7p2p2 结果对比

## 数据来源
- KPI（均值）：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 远端 self-check（ubuntu-zt）
- 命令：
  - `python train.py --self-check`
  - `python infer.py --self-check`
- 结果：均通过（CUDA 可用，`device_ok=cuda:0`）。

### 2) GlobalCNN 单测
- 命令：`conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`
- 结果：`3 passed`
- 覆盖点：
  - `CNNQNetwork` 两种骨干（legacy/globalcnn）前向输出维度正确。
  - `DQNFamilyAgent` 的 GlobalCNN checkpoint 保存/加载可用。
  - 旧 checkpoint（缺失 GlobalCNN 参数）可兼容加载。

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 15.8975 | 10.1250 | 0.054351 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 24.1420 | 17.4500 | 0.110135 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.333 | 44.2185 | 28.3000 | 0.104250 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 通过 | 通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p1` 的趋势对照（仅作 smoke 门决策参考）
- `v7p1`（runs=5）CNN 关键指标：short/mid/long `success_rate=1.00/1.00/1.00`。
- `v7p2p2`（runs=3）CNN 关键指标：short/mid/long `success_rate=0.667/0.333/0.333`。
- 结论：`v7p2p2` 在可达率上明显退化，不满足“明确收益”条件。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）。
- long（runs=20）：`N/A`（本轮仅 smoke）。
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=1`，`collision=1`，`timeout=1`
  - long：`reached=1`，`timeout=2`
  - 合计：`reached=4`，`timeout=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：`success_rate` 在 short/mid/long 全部落后于基线，且 mid/long 的路径时间与曲率无优势。
- 处理：`v7p2p2` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。

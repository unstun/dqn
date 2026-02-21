# v7p2p3 结果对比

## 数据来源
- KPI（均值）：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 远端 self-check（ubuntu-zt）
- 命令：
  - `python train.py --self-check`
  - `python infer.py --self-check`
- 结果：均通过（CUDA 可用，`device_ok=cuda:0`）。

### 2) `globalcnn_fusion` 单测
- 命令：`conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`
- 结果：`3 passed`
- 覆盖点：
  - `CNNQNetwork` 三种骨干（legacy/globalcnn/globalcnn_fusion）前向输出维度正确；
  - `DQNFamilyAgent` 对 `globalcnn/globalcnn_fusion` checkpoint 保存/加载可用；
  - 旧 checkpoint（缺失新字段）兼容加载。

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 13.1250 | 8.2000 | 0.005843 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.333 | 64.7312 | 48.2000 | 0.202941 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 48.5295 | 30.9000 | 0.169235 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 通过 | 通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p2p2` 的趋势对照（仅作 smoke 门决策参考）
- `v7p2p2`（runs=3）CNN：short/mid/long `SR=0.667/0.333/0.333`，`path_time_s=10.125/17.450/28.300`。
- `v7p2p3`（runs=3）CNN：short/mid/long `SR=0.333/0.333/0.667`，`path_time_s=8.200/48.200/30.900`。
- 结论：long SR 局部提升，但 short SR 明显下降且 mid 时间/路径严重恶化，不满足“明确收益”。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）。
- long（runs=20）：`N/A`（本轮仅 smoke）。
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=1`，`timeout=1`，`collision=1`
  - mid：`reached=1`，`timeout=2`
  - long：`reached=2`，`timeout=1`
  - 合计：`reached=4`，`timeout=4`，`collision=1`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：`success_rate` 在 short/mid/long 全部低于基线，且 mid/long 的路径长度、时间与曲率明显劣化。
- 处理：`v7p2p3` 失败归档，不进入 full（`runs=20`），主线保持 `v7p1`。

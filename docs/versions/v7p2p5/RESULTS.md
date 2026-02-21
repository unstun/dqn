# v7p2p5 结果对比

## 数据来源
- KPI（均值）：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626/table2_kpis_raw.csv`
- 运行口径：`episodes=150`（train smoke）+ `runs=3`（infer short/mid/long）。

## 代码级验证结果

### 1) 远端 self-check（ubuntu-zt）
- 命令：
  - `python train.py --self-check`
  - `python infer.py --self-check`
- 结果：均通过（CUDA 可用，`device_ok=cuda:0`）。

### 2) 本地单测（融合归一化新增后）
- 命令：`conda run -n ros2py310 python -m pytest tests/test_globalcnn_network.py -v`
- 结果：`4 passed`
- 覆盖点：
  - `legacy/globalcnn/globalcnn_fusion` 前向输出维度正确；
  - `globalcnn_fusion` 的空间先验 + 融合归一化接线可用；
  - `DQNFamilyAgent` 对新参数 checkpoint roundtrip 可用；
  - 旧 checkpoint（缺失新字段）兼容加载。

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m |
|---|---|---:|---:|---:|---:|
| short | CNN-DDQN | 0.333 | 31.5744 | 17.1500 | 0.191482 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 |
| mid | CNN-DDQN | 0.667 | 42.0740 | 28.1500 | 0.128448 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 |
| long | CNN-DDQN | 0.667 | 89.4701 | 51.0000 | 0.193102 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 不通过 | 不通过 | 不通过 | 未通过 |

## 相对 `v7p2p4` 的趋势对照（仅作 smoke 门决策参考）
- `v7p2p4`（runs=3）CNN：short/mid/long `SR=0.667/0.667/1.000`，`path_time_s=10.725/16.550/28.200`。
- `v7p2p5`（runs=3）CNN：short/mid/long `SR=0.333/0.667/0.667`，`path_time_s=17.150/28.150/51.000`。
- 结论：本轮整体退化，不满足“明确收益”条件。

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 smoke）。
- long（runs=20）：`N/A`（本轮仅 smoke）。
- 结论：本轮未进入最终门槛评测。

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=1`，`timeout=2`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=2`，`timeout=1`
  - 合计：`reached=5`，`timeout=4`
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：success_rate、path_time_s、avg_path_length 对基线均无优势，且相对 `v7p2p4` 明显退化。
- 处理：`v7p2p5` 失败归档，保持当前代码继续前向迭代，不执行代码回退。

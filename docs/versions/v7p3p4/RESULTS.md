# v7p3p4 结果对比

## 数据来源
- KPI（均值）：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_raw.csv`
- 运行口径：infer smoke（固定 `v7p3p2` checkpoint，不重训）+ `runs=3`（short/mid/long）。

## 代码级验证结果

### 1) 单测
- 命令（期望）：`conda run -n ros2py310 python -m pytest tests/test_v7p3p2_turn_aware_topk.py -q`
- 结果：**未执行**（当前 `ros2py310` 环境缺少 `pytest`：`No module named pytest`）。

### 2) 最小自检
- 远端（`ubuntu-zt`）：
  - `ssh ubuntu-zt "... python train.py --self-check"`
  - `ssh ubuntu-zt "... python infer.py --self-check"`
- 结果：通过（CUDA 可用）

## short/mid/long 指标（CNN vs Hybrid A*-MPC）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 22.7499 | 13.2250 | 0.164133 | 0.192 | 0.192 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 0.667 | 36.8081 | 20.7000 | 0.190047 | 0.470 | 0.470 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 1.000 | 71.4983 | 38.1000 | 0.171134 | 0.327 | 0.327 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

## 门槛检查（三条不等式）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | 不通过 | 不通过 | 不通过 | 未通过 |
| mid | 不通过 | 不通过 | 不通过 | 未通过 |
| long | 通过（=） | 不通过 | 不通过 | 未通过 |

## `failure_reason` 分布（来自 `table2_kpis_raw.csv`）
- CNN-DDQN：
  - short：`reached=2`，`timeout=1`
  - mid：`reached=2`，`timeout=1`
  - long：`reached=3`
  - 合计：`reached=7`，`timeout=2`（`collision=0`）
- Hybrid A*-MPC：
  - short：`reached=3`
  - mid：`reached=3`
  - long：`reached=3`
  - 合计：`reached=9`

## 最终门槛口径状态（short/long 各 runs=20）
- short（runs=20）：`N/A`（本轮仅 infer smoke）
- long（runs=20）：`N/A`（本轮仅 infer smoke）
- 结论：本轮未进入最终门槛评测。

## smoke 门结论（go/no-go）
- 结论：**NO-GO**。
- 原因：尽管 safe fallback 补丁消除了 `collision` 并恢复 long `success_rate=1.0`，但 short/mid `success_rate` 仍显著落后 baseline，且 `avg_path_length/path_time_s/avg_curvature_1_m` 明显更差，不满足“全面超过 baseline”的目标。

## 附：推理侧 long-only 压路径消融（固定同一 checkpoint，不重训）

> 目标：在 `success_rate≈1.0` 前提下压 `avg_path_length`（优先级最高）。  
> 结论：本轮三组参数过于激进，**全部 SR=0**（`collision/timeout`），无法进入“压路径”对比阶段。

### 数据来源
- S1：`runs/v7p3p4_pathlen_sweep_long_S1/20260222_172355/table2_kpis_mean_raw.csv`
- S2：`runs/v7p3p4_pathlen_sweep_long_S2/20260222_172426/table2_kpis_mean_raw.csv`
- S3：`runs/v7p3p4_pathlen_sweep_long_S3/20260222_172453/table2_kpis_mean_raw.csv`

### long 套件结果汇总（CNN-DDQN）

| sweep | 关键参数（infer） | success_rate | argmax_inadmissible_rate | fallback_rate | `failure_reason` 分布 |
|---|---|---:|---:|---:|---|
| S1 | `adm_h=10, min_od=0.0, min_prog=-0.2, turn_pen=0.0` | 0.000 | 0.240 | 0.174 | `collision=3` |
| S2 | `adm_h=15, min_od=0.0, min_prog=-0.2, turn_pen=0.0` | 0.000 | 0.530 | 0.523 | `timeout=2, collision=1` |
| S3 | `adm_h=10, min_od=0.0, min_prog=0.0, turn_pen=0.0` | 0.000 | 0.287 | 0.149 | `collision=2, timeout=1` |

### 解释（针对“路径最短优先级最高”）
- 这三组 sweep 的共同点是将 `forest_min_od_m`（最小净空阈值）降到 `0.0`，导致 long 套件出现大量 `collision`，**根本无法完成到达**，因此不具备“路径更短”的可比性。
- 下一步更稳妥的压路径方向应当优先从 **不改变安全判定** 的参数入手（例如只调 `forest_topk_turn_penalty` / 适度调 `forest_adm_horizon`），先确保 `success_rate=1.0` 再讨论 `avg_path_length`。

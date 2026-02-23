# v8p3 结果对比（smoke）

## 1. 数据来源（待回填）

- smoke（episodes=150, runs=3）：
  - `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153/table2_kpis_mean_raw.csv`
- 回归（固定 short collision pair，runs=1）：
  - shielded/hybrid：`runs/v8p3_short_collision_pair_regression/20260223_124513/table2_kpis_mean_raw.csv`
  - strict-argmax（诊断）：`runs/v8p3_short_collision_pair_regression_strict/20260223_124959/table2_kpis_mean_raw.csv`

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地）

### 单元测试
- `conda run -n ros2py310 python -m pytest -q`
- 结果：`PASS`（本地）

## 3. 指标（待回填）

### 回归（fixed pair，runs=1）

> 说明：本回归用例来自 `v8p2` smoke 的 short 可复现 collision pair（`start_xy=(30,241) -> goal_xy=(119,82)`）。

| regime | suite | SR | L | T | inad | fb | failure_reason | run_dir |
|---|---|---:|---:|---:|---:|---:|---|---|
| shielded/hybrid | short | 0.0 | 18.035 | 10.30 | 0.184 | 0.184 | collision | `runs/v8p3_short_collision_pair_regression/20260223_124513` |
| strict-argmax（诊断） | short | 0.0 | 9.570 | 5.55 | 0.270 | 0.000 | collision | `runs/v8p3_short_collision_pair_regression_strict/20260223_124959` |

注：
- `L` = `avg_path_length`（m，越小越好）
- `T` = `path_time_s`（s，越小越好）
- `inad` = `argmax_inadmissible_rate`（诊断）
- `fb` = `fallback_rate`（诊断；strict 下应为 0）

### short/mid/long（runs=3，mean）
（来源：`runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153/table2_kpis_mean_raw.csv`）

| suite | algo | SR | L | T | inad | fb |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 1.000 | 17.1597 | 12.6833 | 0.203 | 0.203 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | N/A | N/A |
| mid | CNN-DDQN | 0.667 | 24.9649 | 16.9000 | 0.237 | 0.237 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | N/A | N/A |
| long | CNN-DDQN | 0.667 | 51.7159 | 34.5750 | 0.439 | 0.439 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | N/A | N/A |

### `failure_reason` 分布
- smoke（runs=3）：
  - short：`reached=3`
  - mid：`reached=2, collision=1`
  - long：`reached=2, timeout=1`
- 回归（fixed pair，runs=1）：
  - shielded/hybrid：`collision=1`
  - strict-argmax（诊断）：`collision=1`

## 4. 门槛检查（smoke，待回填）
- 回归：`FAIL`（collision 仍可复现）
- smoke：`FAIL`（mid `collision=1/3`；long `timeout=1/3`；未满足 `SR≈1.0` 前提）

## 5. 结论（go/no-go，待回填）
- 当前结论：`NO-GO`（smoke 未过门：mid 仍出现 collision，long 仍出现 timeout；下一轮进入 `v8p4`，优先做“最后兜底永不返回碰撞动作/分级降阶 horizon”的 safety 修复，再跑 smoke。）

# v8p3 结果对比（smoke）

## 1. 数据来源（待回填）

- smoke（episodes=150, runs=3）：N/A
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
- N/A（待 smoke）

### `failure_reason` 分布
- 回归（fixed pair，runs=1）：
  - shielded/hybrid：`collision=1`
  - strict-argmax（诊断）：`collision=1`

## 4. 门槛检查（smoke，待回填）
- 回归：`FAIL`（collision 仍可复现）
- smoke：N/A

## 5. 结论（go/no-go，待回填）
- 当前结论：`NO-GO`（v8p3 仅修复了“min_od_m 筛空时的 collision-first fallback”，但 fixed pair 的 collision 未消除；需要更强的 safety 兜底/屏蔽策略或训练侧修复后再进入 smoke/full。）

# v8p4 结果对比（回归 FAIL；smoke 未跑）

## 1. 数据来源

- 回归（fixed pairs，infer-only）：
  - pairs：`configs/pairs_v8p3_smoke_failures.json`
  - `run_dir`: `runs/v8p4_smoke_failures_regression/20260223_142739`
  - `kpi_mean_raw`: `runs/v8p4_smoke_failures_regression/20260223_142739/table2_kpis_mean_raw.csv`
  - `kpi_raw`: `runs/v8p4_smoke_failures_regression/20260223_142739/table2_kpis_raw.csv`
- smoke（episodes=150, runs=3）：
  - `run_dir`: `N/A`（待运行）

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地）

### 单元测试
- `conda run -n ros2py310 python -m pytest -q tests/test_v8p4_fallback_short_rollout_h1.py`
- 结果：`PASS`（本地）

## 3. 指标

### 回归（fixed pairs）

> 说明：本回归一次跑了 `envs=[mid,long]`，每个 suite 使用 `runs=2` 串行加载 `configs/pairs_v8p3_smoke_failures.json` 中的两对（按 `run_idx` 排序）。

| suite | run_idx | start_xy -> goal_xy | SR | L | T | inad | fb | failure_reason | run_dir |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| mid | 0 | (103,260) -> (318,124) | 0.0 | 26.6538 | 16.90 | 0.169 | 0.169 | collision | `runs/v8p4_smoke_failures_regression/20260223_142739` |
| mid | 1 | (30,32) -> (329,327) | 0.0 | 36.5158 | 60.00 | 0.732 | 0.732 | timeout | `runs/v8p4_smoke_failures_regression/20260223_142739` |
| long | 0 | (103,260) -> (318,124) | 0.0 | 26.6538 | 16.90 | 0.169 | 0.169 | collision | `runs/v8p4_smoke_failures_regression/20260223_142739` |
| long | 1 | (30,32) -> (329,327) | 0.0 | 36.5158 | 60.00 | 0.732 | 0.732 | timeout | `runs/v8p4_smoke_failures_regression/20260223_142739` |

### `failure_reason` 分布（回归，runs=2）
- mid：`collision=1, timeout=1`
- long：`collision=1, timeout=1`

注：
- `L` = `avg_path_length`（m，越小越好）
- `T` = `path_time_s`（s，越小越好）
- `inad` = `argmax_inadmissible_rate`（诊断）
- `fb` = `fallback_rate`（诊断）

### short/mid/long（smoke，mean）

| suite | algo | SR | L | T | inad | fb |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | N/A | N/A | N/A | N/A | N/A |
| short | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A |
| mid | CNN-DDQN | N/A | N/A | N/A | N/A | N/A |
| mid | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A |
| long | CNN-DDQN | N/A | N/A | N/A | N/A | N/A |
| long | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A |

## 4. 门槛检查

- 回归：`FAIL`（fixed pairs 仍出现 `collision/timeout`）
- smoke：`N/A`（未跑）

## 5. 结论（go/no-go）

- 当前结论：`NO-GO`（回归失败；暂不进入 smoke，进入 `v8p5` 继续迭代）

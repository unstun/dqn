# v8p1 结果对比（待回填）

## 1. 数据来源（待回填）

### 1) infer-only smoke（固定 v7p1 checkpoint）
- `grid4`：`N/A`
- `euclid`（对照）：`N/A`

### 2) train+infer smoke（episodes=150, runs=3）
- train：`N/A`
- infer：`N/A`

## 2. 代码级验证结果

### 最小自检
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（本地已通过；远端待回填）

## 3. short/mid/long 指标（待回填）

> 说明：请将 `table2_kpis_mean_raw.csv` 的均值行回填到下表，并在 `docs/versions/v8p1/runs/README.md` 记录对应路径。

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| short | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |
| mid | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| mid | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |
| long | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| long | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |

## 4. 门槛检查（短/长 hard gate）

| 套件 | `SR(CNN)>=SR(Hybrid)` | `Path(CNN)<Path(Hybrid)` | `Time(CNN)<Time(Hybrid)` | 结论 |
|---|---|---|---|---|
| short | N/A | N/A | N/A | N/A |
| long | N/A | N/A | N/A | N/A |

## 5. 结论（go/no-go）
- 结论：`N/A`
- 证据路径：
  - `N/A`


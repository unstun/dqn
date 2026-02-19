# v6p2p2 结果

## 数据来源
- smoke 网格汇总（9 组）：
  - `runs/repro_20260219_v6p2p2_reward_sweep/smoke_summary_latest.csv`
- smoke 排名原始 KPI：
  - 见 `docs/versions/v6p2p2/runs/README.md` 中每组 `kpi_csv`
- 20-run 复核：
  - `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433/table2_kpis_mean_raw.csv`
  - `runs/repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20/20260219_123827/table2_kpis_mean_raw.csv`

## 一、smoke 网格排名（2 套件平均）

| rank | k_t | k_delta | success_rate | avg_path_length | avg_curvature_1_m | path_time_s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.06 | 1.5 | 0.6670 | 38.4131 | 0.166574 | 30.8375 |
| 2 | 0.06 | 2.2 | 0.6665 | 43.0736 | 0.170700 | 36.6334 |
| 3 | 0.10 | 0.8 | 0.5000 | 30.0408 | 0.078748 | 18.9625 |
| 4 | 0.14 | 0.8 | 0.5000 | 40.9767 | 0.220658 | 26.6625 |
| 5 | 0.14 | 2.2 | 0.3335 | 13.5430 | 0.193443 | 10.5250 |
| 6 | 0.14 | 1.5 | 0.3335 | 15.3486 | 0.361034 | 12.7500 |
| 7 | 0.10 | 2.2 | 0.3335 | 16.1448 | 0.161578 | 12.6500 |
| 8 | 0.10 | 1.5 | 0.3330 | 37.5773 | 0.096391 | 24.3750 |
| 9 | 0.06 | 0.8 | 0.3330 | 49.1152 | 0.265649 | 33.2250 |

说明：rank 规则为 `success_rate` 优先，其次 `avg_path_length`、`avg_curvature_1_m`、`path_time_s`。

## 二、20-run 复核（候选对照）

### 1) `k_t=0.10, k_delta=0.8`
- short（CNN）：SR=`0.75`，len=`20.5956`，curv=`0.232669`，time=`16.1633`
- long（CNN）：SR=`0.55`，len=`59.1761`，curv=`0.246381`，time=`42.8682`
- 2 套件平均（CNN）：SR=`0.65`，len=`39.88585`，curv=`0.239525`，time=`29.51575`
- failure_reason 分布（CNN）
  - short：`reached=15`, `timeout=4`, `collision=1`
  - long：`reached=11`, `timeout=8`, `collision=1`

### 2) `k_t=0.06, k_delta=1.5`
- short（CNN）：SR=`0.45`，len=`20.8154`，curv=`0.203479`，time=`17.0889`
- long（CNN）：SR=`0.30`，len=`50.8996`，curv=`0.176859`，time=`33.8083`
- 2 套件平均（CNN）：SR=`0.375`，len=`35.8575`，curv=`0.190169`，time=`25.4486`
- failure_reason 分布（CNN）
  - short：`reached=9`, `timeout=10`, `collision=1`
  - long：`reached=6`, `timeout=14`

## 三、参数结论（针对“更短 + 更平滑”）
- 在本轮数据中，`k_t=0.06, k_delta=1.5` 相比 `k_t=0.10, k_delta=0.8`：
  - 平均路径更短（`35.8575 < 39.88585`）
  - 平均曲率更低（`0.190169 < 0.239525`）
  - 平均路径时间更低（`25.4486 < 29.51575`）
- 因此，若目标明确是“路径更短、曲线更平滑”，本版建议参数：
  - **`forest_reward_k_t=0.06`，`forest_reward_k_delta=1.5`**

## 四、最终门槛检查（说明）
- 本轮完成的是：`smoke 网格 + infer20 复核`。
- 未执行 `train300 + short/long runs=20` 的完整闭环确认，故最终门槛结论记为：`N/A（待 full）`。
- 以 20-run 复核看，候选参数相对 `Hybrid A*-MPC` 三条门槛均未满足（short/long 均存在 SR、len、time 回退）。

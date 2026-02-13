# v6p2 - 结果

> 说明：本版为“仓库规范化/自检入口”补齐，不新增训练/推理实验；KPI 与门槛检查沿用 v6p1 的 fixed pairs20 结果。

## 代表性正式指标（继承 v6p1，fixed pairs20 v1）

### short（full runs=20）
- KPI：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.70`，avg_path_length=`15.1378`，path_time_s=`9.9179`，argmax_inad=`0.260`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`15.5982`，path_time_s=`9.2342`
- failures：`reached=14, timeout=3, collision=3`

### long（full runs=20）
- KPI：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.95`，avg_path_length=`34.3502`，path_time_s=`21.5474`，argmax_inad=`0.270`
  - Hybrid A*-MPC：SR=`0.90`，avg_path_length=`32.8275`，path_time_s=`17.7861`
- failures：`reached=19, timeout=1`
- timeout run_idx：`[13]`

## 门槛检查（继承 v6p1，对标 Hybrid A*-MPC）

### short
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.70 >= 0.95` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `15.1378 < 15.5982` → **PASS**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `9.9179 < 9.2342` → **FAIL**

### long
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.95 >= 0.90` → **PASS**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `34.3502 < 32.8275` → **FAIL**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `21.5474 < 17.7861` → **FAIL**

### 最终门槛状态
- `未通过`


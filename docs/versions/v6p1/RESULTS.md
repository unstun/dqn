# v6p1 - 结果

> 说明：本版本主目标是降低 long/mid 的 `timeout`，但最终研究门槛仍按 short/long 双套件、各 `runs=20` 的硬口径检查。

## 代表性正式指标（fixed pairs20 v1）

### long（full runs=20，v6p1 最终采用）
- KPI：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.95`，avg_path_length=`34.3502`，path_time_s=`21.5474`，argmax_inad=`0.270`
  - Hybrid A*-MPC：SR=`0.90`，avg_path_length=`32.8275`，path_time_s=`17.7861`
- failures：`reached=19, timeout=1`
- timeout run_idx：`[13]`

### mid（full runs=20，回归检查）
- KPI：`runs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1/20260212_003901/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.95`，avg_path_length=`23.2018`，path_time_s=`15.3553`，argmax_inad=`0.244`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`21.6854`，path_time_s=`12.6553`
- failures：`reached=19, collision=1`

### short（full runs=20，迁移失败示例；不建议采用）
- KPI：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.70`，avg_path_length=`15.1378`，path_time_s=`9.9179`，argmax_inad=`0.260`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`15.5982`，path_time_s=`9.2342`
- failures：`reached=14, timeout=3, collision=3`
- 备注：short 推荐仍沿用 v6：`configs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1.json`（SR=`0.90`，`reached=18, collision=2`）。

## 对比 v6（同 checkpoint，fixed pairs20）
- long：v6 `timeout=5/20, SR=0.70` → v6p1 `timeout=1/20, SR=0.95`
- mid：v6 `timeout=2/20, SR=0.90` → v6p1 `timeout=0/20, SR=0.95`（但出现 `collision=1/20`）
- short：v6 `SR=0.90` → v6p1（long-tuned gating）`SR=0.70`（显著回退）

## 门槛检查（对标 Hybrid A*-MPC）

### short（v6p1 short）
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.70 >= 0.95` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `15.1378 < 15.5982` → **PASS**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `9.9179 < 9.2342` → **FAIL**

### long（v6p1 long）
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.95 >= 0.90` → **PASS**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `34.3502 < 32.8275` → **FAIL**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `21.5474 < 17.7861` → **FAIL**

### 最终门槛状态
- `未通过`

## smoke 选参（long timeout union10）
- pairs：`configs/repro_20260211_forest_a_pairs_long10_timeout_union_v5v6_v1.json`
- 选中的组合：`forest_adm_horizon=35` + `forest_min_od_m=0.002`（`forest_min_progress_m=0.01`）
- smoke run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225`
  - CNN-DDQN failures：`reached=10`（timeout/collision 均为 0）


# v6 - 结果

## 代表性正式指标（fixed pairs20 v1）

> 说明：本版本主目标为降低 `timeout`，但最终结论仍按 short/long 双套件、各 `runs=20` 的门槛口径汇报。

### short（full runs=20）
- KPI：`runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.90`，avg_path_length=`15.9326`，path_time_s=`11.4833`，argmax_inad=`0.227`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`15.5982`，path_time_s=`9.2342`

### mid（full runs=20）
- KPI：`runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.90`，avg_path_length=`23.4322`，path_time_s=`16.9944`，argmax_inad=`0.299`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`21.6854`，path_time_s=`12.6553`

### long（full runs=20）
- KPI：`runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.70`，avg_path_length=`34.2745`，path_time_s=`19.775`，argmax_inad=`0.281`
  - Hybrid A*-MPC：SR=`0.90`，avg_path_length=`32.8275`，path_time_s=`17.7861`

## 对比 v5 baseline（同一 checkpoint + v5 默认推理参数）
- short baseline：`runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.75`，timeout=`1/20`
- mid baseline：`runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.80`，timeout=`4/20`
- long baseline：`runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706/table2_kpis_mean_raw.csv`
  - CNN-DDQN：SR=`0.65`，timeout=`6/20`

## 门槛检查（对标 Hybrid A*-MPC）
### short
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.90 >= 0.95` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `15.9326 < 15.5982` → **FAIL**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `11.4833 < 9.2342` → **FAIL**

### long
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.70 >= 0.90` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `34.2745 < 32.8275` → **FAIL**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `19.775 < 17.7861` → **FAIL**

### 最终门槛状态
- `未通过`

## 失败分析（failure_reason 分布）
### v6（full runs=20）
- short：`reached=18, collision=2, timeout=0`
- mid：`reached=18, timeout=2`
- long：`reached=14, collision=1, timeout=5`

### long 超时迁移（pair-level）
- v5 baseline long timeout run_idx：`[7, 10, 11, 15, 16, 18]`
- v6 tuned long timeout run_idx：`[0, 2, 14, 15, 19]`
- 说明：本版 long 的 `timeout` 总数小幅下降，但失败样本发生迁移；需要进一步 sweep `forest_min_progress_m` / `forest_min_od_m` 以减少“修复 A 引入 B”的现象。

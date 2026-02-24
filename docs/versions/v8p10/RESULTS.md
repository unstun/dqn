# v8p10 结果对比（待跑：progress-dist clearance sweep smoke → full gate C）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径（计划）

- 推理侧 sweep smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p10_infer_sweep_short_smoke.json`
  - long profile：`configs/repro_20260224_v8p10_infer_sweep_long_smoke.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir（w_clearance=0.0, ranking=progress_q）：`runs/v8p10_infer_sweep_short_pairs3_smoke/20260224_134521`
  - long run_dir（w_clearance=0.0, ranking=progress_q）：`runs/v8p10_infer_sweep_long_pairs3_smoke/20260224_134539`

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) sweep smoke 结果（runs=3，mean）

baseline（Hybrid A*-MPC）仍显著短于 RL；v8p10 相对 v8p9 在 long 上有回落，但距离“短于 baseline”仍差距很大。

### 2.1 short（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=16.3207`，`path_time_s=9.4667`，`success_rate=1.0`

RL（CNN-DDQN, hybrid/shielded）：`avg_path_length=19.6757`，`path_time_s=11.7833`，`success_rate=1.0`

failure_reason（RL）：`reached=3/3`

### 2.2 long（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=32.2801`，`path_time_s=17.4333`，`success_rate=1.0`

RL（CNN-DDQN, hybrid/shielded）：`avg_path_length=46.5236`，`path_time_s=25.8167`，`success_rate=1.0`

failure_reason（RL）：`reached=3/3`

**阶段结论（smoke）**：
- v8p10（`dijkstra8_nocorner` + `w_clearance=0.0` + `progress_q`）可在 fixed pairs3 上保持 SR=1.0（short/long）。
- 但 RL 的 `avg_path_length/path_time_s` 仍明显落后 baseline（尤其 long），因此不建议直接上 full gate C；需要继续做 `w_clearance` sweep（0.5/1.0/2.0）与 ranking/阈值联动搜索，或进一步进入训练侧改动。

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）

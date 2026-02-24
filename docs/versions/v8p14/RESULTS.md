# v8p14 结果对比（infer-only sweep 已跑：NO-GO；不建议 full gate C）

## 1) 关键工件路径

- infer sweep（long，fixed pairs3，runs=3）：
  - cfg（mp=-0.02）：`configs/repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg002.json`
  - cfg（mp=-0.05）：`configs/repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg005.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - run_dir（mp=-0.02）：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg002/20260224_174532`
  - run_dir（mp=-0.05）：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/20260224_174542`

## 2) sweep 结果（mean）

### long（mp=-0.02）

- KPI：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg002/20260224_174532/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=38.1854`，`path_time_s=22.8667`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.9053`，`path_time_s=+5.4334`

### long（mp=-0.05）

- KPI：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/20260224_174542/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=38.1797`，`path_time_s=20.7333`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.8996`，`path_time_s=+3.3000`

结论：
- `forest_min_progress_m` 设为小负数后，long 的 `SR` 恢复到 `1.0`（不再 timeout），符合“恢复成功率”的预期。
- 但 long 的 `avg_path_length/path_time_s` 仍显著落后 baseline（Hybrid A*-MPC），且未优于 v8p11 的 long；本版判定：**NO-GO（不建议 full gate C）**。

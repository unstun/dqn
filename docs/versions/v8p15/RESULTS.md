# v8p15 结果对比（infer-only sweep 已跑：NO-GO；不建议 full gate C）

## 1) 关键工件路径

- infer sweep（long，fixed pairs3，runs=3）：
  - cfg（sigma=0.2）：`configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p2.json`
  - cfg（sigma=0.3）：`configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p3.json`
  - cfg（sigma=0.5）：`configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p5.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - run_dir（sigma=0.2）：`runs/v8p15_sweep_long_pairs3_sigma0p2/20260224_180451`
  - run_dir（sigma=0.3）：`runs/v8p15_sweep_long_pairs3_sigma0p3/20260224_180520`
  - run_dir（sigma=0.5）：`runs/v8p15_sweep_long_pairs3_sigma0p5/20260224_180550`

## 2) sweep 结果（mean）

### long（sigma=0.2）

- KPI：`runs/v8p15_sweep_long_pairs3_sigma0p2/20260224_180451/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=48.1380`，`path_time_s=25.7333`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+15.8579`，`path_time_s=+8.3000`

### long（sigma=0.3）

- KPI：`runs/v8p15_sweep_long_pairs3_sigma0p3/20260224_180520/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=37.9271`，`path_time_s=20.3333`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.6470`，`path_time_s=+2.9000`

### long（sigma=0.5）

- KPI：`runs/v8p15_sweep_long_pairs3_sigma0p5/20260224_180550/table2_kpis_mean_raw.csv`
- CNN-DDQN：`success_rate=1.0`，`avg_path_length=38.1797`，`path_time_s=20.7333`
- Hybrid A*-MPC：`success_rate=1.0`，`avg_path_length=32.2801`，`path_time_s=17.4333`
- Δ（RL - baseline）：`avg_path_length=+5.8996`，`path_time_s=+3.3000`

结论：
- `sigma=0.3` 为本轮最优（相对 `sigma=0.5` long 路径更短、时间更快），且 `SR=1.0`；但相比 baseline 仍明显落后，未达“路径比 baseline 短”的目标。
- 本版判定：**NO-GO（不建议 full gate C）**。

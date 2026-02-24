# v8p16 — RESULTS（待补）

## 0) 评测口径（硬约束）

- **fixed pairs + baseline 同跑**：避免 sample drift；baseline 统一为 `Hybrid A*-MPC`。
- 禁止作弊：不改 `goal_tolerance_m`（终点容差）与 stop 阈值。
- 口径命名：本版默认 `shielded/hybrid` 推理（不要写成 `strict-argmax`）。
- 运行位置：`ssh ubuntu-zt` 不可用（2026-02-24），本轮按“本地回落流程”在本机执行。

## 1) 输入与产出路径（待补）

- 训练 smoke：
  - config：`configs/repro_20260224_v8p16_train_smoke.json`
  - run_dir：`runs/v8p16/train_20260224_020118`
  - checkpoint：`runs/v8p16/train_20260224_020118/models/forest_a/cnn-ddqn.pt`
- 推理 smoke（short/long，pairs3）：
  - short config：`configs/repro_20260224_v8p16_infer_smoke_short.json`
  - long config：`configs/repro_20260224_v8p16_infer_smoke_long.json`
  - short run_dir：`runs/v8p16_infer_smoke_short_pairs3/20260224_023632`
  - long run_dir：`runs/v8p16_infer_smoke_long_pairs3/20260224_023542`
  - `table2_kpis_mean_raw.csv`：
    - short：`runs/v8p16_infer_smoke_short_pairs3/20260224_023632/table2_kpis_mean_raw.csv`
    - long：`runs/v8p16_infer_smoke_long_pairs3/20260224_023542/table2_kpis_mean_raw.csv`

## 2) KPI（pairs3，runs=3）（待补）

> 填写时请同时给出 CNN 与 baseline 的 `avg_path_length` / `path_time_s` / `success_rate`，并明确 short/long。

- short（forest_a::short）：
  - CNN-DDQN：`SR=1.000`，`avg_path_length=18.6356`，`path_time_s=11.2667`
  - Hybrid A*-MPC：`SR=1.000`，`avg_path_length=16.3207`，`path_time_s=9.4667`
- long（forest_a::long）：
  - CNN-DDQN：`SR=1.000`，`avg_path_length=39.9377`，`path_time_s=25.4167`
  - Hybrid A*-MPC：`SR=1.000`，`avg_path_length=32.2801`，`path_time_s=17.4333`

## 3) 门槛检查（最终 gate 用 runs=20）

- short：`success_rate(CNN) >= success_rate(baseline)`：N/A
- long：`success_rate(CNN) >= success_rate(baseline)`：N/A
- short：`avg_path_length(CNN) < avg_path_length(baseline)`：N/A
- long：`avg_path_length(CNN) < avg_path_length(baseline)`：N/A
- short：`path_time_s(CNN) < path_time_s(baseline)`：N/A
- long：`path_time_s(CNN) < path_time_s(baseline)`：N/A

## 4) 结论（待补）

- 结论：NO-GO（smoke：SR=1.0 可维持，但 short/long 的 `avg_path_length` 与 `path_time_s` 均落后 baseline；long detour 反而更大）。
- 证据路径：
  - short：`runs/v8p16_infer_smoke_short_pairs3/20260224_023632/table2_kpis_mean_raw.csv`
  - long：`runs/v8p16_infer_smoke_long_pairs3/20260224_023542/table2_kpis_mean_raw.csv`
- 下一步：进入 C 线（结构/算法变种），优先选择不引入新依赖的 DQN 变种做 smoke。

# v8p16 — RESULTS（待补）

## 0) 评测口径（硬约束）

- **fixed pairs + baseline 同跑**：避免 sample drift；baseline 统一为 `Hybrid A*-MPC`。
- 禁止作弊：不改 `goal_tolerance_m`（终点容差）与 stop 阈值。
- 口径命名：本版默认 `shielded/hybrid` 推理（不要写成 `strict-argmax`）。

## 1) 输入与产出路径（待补）

- 训练 smoke：
  - config：`configs/repro_20260224_v8p16_train_smoke.json`
  - run_dir：N/A
  - checkpoint：N/A
- 推理 smoke（short/long，pairs3）：
  - short config：`configs/repro_20260224_v8p16_infer_smoke_short.json`
  - long config：`configs/repro_20260224_v8p16_infer_smoke_long.json`
  - short run_dir：N/A
  - long run_dir：N/A
  - `table2_kpis_mean_raw.csv`：N/A

## 2) KPI（pairs3，runs=3）（待补）

> 填写时请同时给出 CNN 与 baseline 的 `avg_path_length` / `path_time_s` / `success_rate`，并明确 short/long。

- short（forest_a::short）：
  - CNN-DDQN：`SR=N/A`，`avg_path_length=N/A`，`path_time_s=N/A`
  - Hybrid A*-MPC：`SR=N/A`，`avg_path_length=N/A`，`path_time_s=N/A`
- long（forest_a::long）：
  - CNN-DDQN：`SR=N/A`，`avg_path_length=N/A`，`path_time_s=N/A`
  - Hybrid A*-MPC：`SR=N/A`，`avg_path_length=N/A`，`path_time_s=N/A`

## 3) 门槛检查（最终 gate 用 runs=20）

- short：`success_rate(CNN) >= success_rate(baseline)`：N/A
- long：`success_rate(CNN) >= success_rate(baseline)`：N/A
- short：`avg_path_length(CNN) < avg_path_length(baseline)`：N/A
- long：`avg_path_length(CNN) < avg_path_length(baseline)`：N/A
- short：`path_time_s(CNN) < path_time_s(baseline)`：N/A
- long：`path_time_s(CNN) < path_time_s(baseline)`：N/A

## 4) 结论（待补）

- 结论：N/A
- 若 NO-GO：请写清楚证据路径（`run_dir` + CSV），并给下一步（进入 C 线或做消融）。


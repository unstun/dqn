# v8p10 结果对比（待跑：progress-dist clearance sweep smoke → full gate C）

> 说明：本版目标是最终硬门槛 C（short/long 各 runs=20，fixed pairs20）。在 full gate 结果出来前，所有 smoke 仅用于筛查，不作为最终结论。

## 1) 关键工件路径（计划）

- 推理侧 sweep smoke（fixed pairs3，runs=3）：
  - short profile：`configs/repro_20260224_v8p10_infer_sweep_short_smoke.json`
  - long profile：`configs/repro_20260224_v8p10_infer_sweep_long_smoke.json`
  - short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
  - long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`
  - short run_dir（ranking=progress_q）：
    - w=0.0：`runs/v8p10_infer_sweep_short_pairs3_smoke/20260224_134521`
    - w=0.5：`runs/v8p10_sweep_short_w0p5/20260224_135059`
    - w=1.0：`runs/v8p10_sweep_short_w1p0/20260224_135116`
    - w=2.0：`runs/v8p10_sweep_short_w2p0/20260224_135134`（当前 best）
  - long run_dir（ranking=progress_q）：
    - w=0.0：`runs/v8p10_infer_sweep_long_pairs3_smoke/20260224_134539`
    - w=0.5：`runs/v8p10_sweep_long_w0p5/20260224_134940`
    - w=1.0：`runs/v8p10_sweep_long_w1p0/20260224_135010`
    - w=2.0：`runs/v8p10_sweep_long_w2p0/20260224_135035`（当前 best）

- full gate（C：fixed pairs20；short/long 各 runs=20）：
  - short pairs：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - long pairs：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - short run_dir：`N/A`
  - long run_dir：`N/A`

## 2) sweep smoke 结果（runs=3，mean）

baseline（Hybrid A*-MPC）仍显著短于 RL；v8p10 相对 v8p9 在 long 上有回落，但距离“短于 baseline”仍差距很大。

### 2.1 short（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=16.3207`，`path_time_s=9.4667`，`success_rate=1.0`

| 变体（w_clearance） | RL success_rate | RL avg_path_length | RL path_time_s | 备注 |
|---:|---:|---:|---:|---|
| 0.0 | 1.0 | 19.6757 | 11.7833 | `runs/v8p10_infer_sweep_short_pairs3_smoke/20260224_134521` |
| 0.5 | 1.0 | 19.6702 | 12.3167 | `runs/v8p10_sweep_short_w0p5/20260224_135059` |
| 1.0 | 1.0 | 19.4829 | 12.5000 | `runs/v8p10_sweep_short_w1p0/20260224_135116` |
| 2.0 | 1.0 | **18.7727** | **11.6333** | `runs/v8p10_sweep_short_w2p0/20260224_135134`（当前 best） |

failure_reason（RL）：`reached=3/3`

### 2.2 long（fixed pairs3）

baseline（Hybrid A*-MPC）：`avg_path_length=32.2801`，`path_time_s=17.4333`，`success_rate=1.0`

| 变体（w_clearance） | RL success_rate | RL avg_path_length | RL path_time_s | 备注 |
|---:|---:|---:|---:|---|
| 0.0 | 1.0 | 46.5236 | 25.8167 | `runs/v8p10_infer_sweep_long_pairs3_smoke/20260224_134539` |
| 0.5 | 1.0 | 43.7560 | 23.6167 | `runs/v8p10_sweep_long_w0p5/20260224_134940` |
| 1.0 | 1.0 | 44.9049 | 24.4333 | `runs/v8p10_sweep_long_w1p0/20260224_135010` |
| 2.0 | 1.0 | **41.0694** | **22.0667** | `runs/v8p10_sweep_long_w2p0/20260224_135035`（当前 best） |

failure_reason（RL）：`reached=3/3`

**阶段结论（smoke）**：
- v8p10（`dijkstra8_nocorner` + `progress_q`）可在 fixed pairs3 上保持 SR=1.0（short/long）。
- `w_clearance` 在该 pairs3 子集上确实能显著压缩 RL 的路径与时间（当前 best 为 w=2.0），但仍明显落后 baseline（尤其 long），因此不建议直接上 full gate C；下一步需要继续做 ranking/阈值联动搜索，或进一步进入训练侧改动。

### 2.3 补充消融：`replace_topq`

在 w=2.0 下测试 `replace_topq=0`（不做 top-Q 限制）：
- long：`avg_path_length=41.1538`（略差于 topq=3 的 41.0694），`success_rate=1.0`
- short：`success_rate=0.667`（出现 `timeout=1/3`），NO-GO

结论：本版默认仍保留 `replace_topq=3`。

### 2.4 补充消融：`min_progress_m`

在 w=2.0、`replace_topq=3` 下测试 `min_progress_m=0.02`：
- long：`avg_path_length=41.2647`（略差于默认 0.01 的 41.0694）
- short：`avg_path_length=18.7631`（与默认 0.01 的 18.7727 基本持平）

结论：本版默认仍保留 `min_progress_m=0.01`。

## 3) full gate（C）结果（runs=20，fixed pairs）

- short：`N/A`
- long：`N/A`

门槛检查（C）：
- `N/A`（等待填入 short/long 的 `table2_kpis_mean_raw.csv` 后判定）

# v6p1 - 变更

## 版本意图
- 延续 v6 的 “不改算法代码，仅调推理期 `admissible-action gating`” 路线，重点压 long（并回归检查 mid）的 `timeout`，降低 timeout 迁移现象。

## 相对 v6 的具体变更
- 代码实现：`无`（不修改 `forest_vehicle_dqn/**`）。
- 新增 long `timeout` union 子集（pairs10）用于 smoke sweep：
  - `configs/repro_20260211_forest_a_pairs_long10_timeout_union_v5v6_v1.json`
- 新增 v6p1 long union10 smoke sweep profiles（均开启 traces）：
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_v6params_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varA_minod0_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varB_minprog_neg002_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varA2_minod001_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varC_h20_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varD_minod0_h40_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varD0_minod0_h34_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varD1_minod0_h35_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varE_minod005_h35_v1.json`
  - `configs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1.json`
- 新增 v6p1 full20 profiles：
  - long：`configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json`（用于最终采用）
  - mid：`configs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1.json`（回归检查）
  - short：`configs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1.json`（实测回退，不建议采用；结果仍归档）

## v6p1 最终采用的关键调参（long/mid）
- `forest_adm_horizon`（可采纳性短视滚动步数）：`30 -> 35`
- `forest_min_od_m`（最小净空阈值）：`0.02 -> 0.002`
- `forest_min_progress_m`（最小进度阈值）：保持 `0.01`

## 变更文件清单
- `configs/repro_20260211_forest_a_pairs_long10_timeout_union_v5v6_v1.json`
- `configs/repro_20260211_v6p1_smoke_long_timeout_union10_*.json`
- `configs/repro_20260211_v6p1_timeout_tune_hybrid_{short,mid,long}_pairs20_v1.json`
- `docs/versions/v6p1/README.md`
- `docs/versions/v6p1/CHANGES.md`
- `docs/versions/v6p1/RESULTS.md`
- `docs/versions/v6p1/runs/README.md`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`


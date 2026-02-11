# v6 - 变更

## 版本意图
- 以 `timeout` 为主瓶颈（尤其 long 套件），在不改算法代码的前提下，通过推理期可采纳性筛选参数（`admissible-action gating`）降低超时占比。

## 相对 v5 的具体变更
- 代码实现：`无`（不修改 `forest_vehicle_dqn/**`）。
- 新增 v6 推理可复现配置（fixed pairs20，hybrid/shielded）：
  - `configs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1.json`
  - `configs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1.json`
  - `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json`
- v6 主要调参（相对 v5 train300 复测默认口径）：
  - `forest_min_progress_m`（可采纳性最小进度阈值）：`-0.02 -> 0.01`
  - `forest_adm_horizon`（可采纳性短视滚动步数）：`20 -> 30`
  - `forest_min_od_m`（可采纳性最小净空阈值）：`0.00 -> 0.02`

## 关键参数快照
- checkpoint models：`runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models`
- 评测样本（fixed pairs20 v1）：
  - short：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - mid：`configs/repro_20260211_forest_a_pairs_mid20_v1.json`
  - long：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
- 推理口径：`hybrid/shielded`（`forest_no_fallback=false`）
- `max_steps=1200`（≈60s）

## 变更文件清单
- `configs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1.json`
- `configs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1.json`
- `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json`
- `docs/versions/v6/README.md`
- `docs/versions/v6/CHANGES.md`
- `docs/versions/v6/RESULTS.md`
- `docs/versions/v6/runs/README.md`


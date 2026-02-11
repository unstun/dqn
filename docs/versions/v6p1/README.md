# v6p1 版本说明

- 版本类型：**Patch（p+1）**
- 研究路线：`CNN-DDQN` 推理主口径：`hybrid/shielded`
- 状态：**未通过（short 套件回退；long 套件 timeout 显著降低）**
- 上一版本：`v6`

## 背景与问题定位
- v6 在 fixed pairs20 下已改善 short/mid 的 `timeout`，但 long 仍有 `timeout=5/20`，且存在 “timeout 样本迁移”（修复一批 timeout 的同时引入新的 timeout pair）。
- long 的 timeout 样本通常伴随更高的 `argmax_inadmissible_rate`（Q 最大动作在可采纳性检查下不合格的比例），提示“策略偏好不可行/不进展动作 → replacement 频繁 → 绕/抖到超时”。

## 本版目标
- **不改算法代码**（仅调 `infer.py` 已支持的参数），在固定同一 checkpoint 的前提下进一步降低 long（并顺带检查 mid）的 `timeout` 占比。
- 口径合规：本版主口径为 `hybrid/shielded`（允许 stop-override + admissibility-gated replacement），不宣称 `strict-argmax`。

## 方法概要（推理期 gating 调参）
1) 构造 long 的 `timeout` union 子集（pairs10），覆盖 v5 baseline 与 v6 tuned 的 timeout 样本，避免继续出现“修复 A 引入 B”。  
2) 在 pairs10 上做 smoke sweep（开启 `--forest-policy-save-traces` 保存每步轨迹 CSV），主要调：
   - `forest_min_od_m`（最小净空阈值）
   - `forest_adm_horizon`（可采纳性短视滚动步数）
3) 选出在 union10 上全通过的组合：`forest_adm_horizon=35` + `forest_min_od_m=0.002`（`forest_min_progress_m=0.01` 保持不变），并在 long fixed pairs20 full20 上复评确认。

## 复现实验配置 / 命令
- long（full runs=20，fixed pairs20）：
  - `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1`
- mid（full runs=20，fixed pairs20；用于回归检查）：
  - `configs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1.json`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1`
- short：**不建议使用本版 gating**（实测回退）；短距仍沿用 v6：
  - `configs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1.json`

## 代表性运行
- long full20：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/table2_kpis_mean_raw.csv`
- long union10 smoke（全通过，用于选参 + traces）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225/traces/`

## 结论 / 下一步
- long：`timeout` 从 v6 的 `5/20` 降到 `1/20`，SR 从 `0.70` 提升到 `0.95`（同 checkpoint + fixed pairs20）。
- mid：`timeout` 从 v6 的 `2/20` 降到 `0/20`，SR 从 `0.90` 提升到 `0.95`（但引入 `collision=1/20`）。
- short：将 long 的 gating 直接迁移到 short 会显著回退（SR=`0.70`）；短距需要单独调参或保持 v6 口径。
- 下一步建议：
  1) 针对 long 剩余 timeout（run_idx=13）单独开 traces 复现定位；
  2) 为 short 套件另做一轮 “短距专用” sweep（目标：减少 collision/timeout 且 SR 不回退）。


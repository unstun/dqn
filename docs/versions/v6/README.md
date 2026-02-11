# v6 版本说明

- 版本类型：**Major (v+1)**
- 研究路线：`CNN-DDQN` 推理主口径：`hybrid/shielded`
- 状态：**未通过**
- 上一版本：`v5`

## 概要
- 目标：在不改算法代码（`forest_vehicle_dqn/**`）的前提下，降低推理阶段 `timeout`（回合跑满 `max_steps` 仍未 `reached`）占比，优先改善 long（以及 mid）套件的超时崩塌。
- baseline（来自 v5 的最新模型复测）：`runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706` 中，`CNN-DDQN` 在 long fixed pairs20 下 `timeout=6/20`（SR=`0.65`）。
- 本版策略：固定同一 checkpoint 输入，仅通过推理期 `admissible-action gating`（可采纳性筛选）参数调优，观察 `timeout` 计数、`success_rate` 与 `argmax_inadmissible_rate` 的变化。
- 结果（full runs=20，fixed pairs20）：
  - short：SR=`0.90`，failures=`reached=18, collision=2, timeout=0`
  - mid：SR=`0.90`，failures=`reached=18, timeout=2`
  - long：SR=`0.70`，failures=`reached=14, collision=1, timeout=5`
- 对比 baseline（同一模型 + v5 默认推理参数）：
  - short：SR `0.75 -> 0.90`，`timeout 1 -> 0`
  - mid：SR `0.80 -> 0.90`，`timeout 4 -> 2`
  - long：SR `0.65 -> 0.70`，`timeout 6 -> 5`

## 方法意图（口径合规）
- `strict-argmax` 仍严格遵守纯 `argmax(Q)` 定义；本版主口径为 `hybrid/shielded`（允许 stop-override + replacement），不宣称 strict。
- 调参项均为 `infer.py` 已支持的 CLI/config 参数，仅改变推理期筛选阈值，不改网络结构/损失/更新规则，避免“换皮不合规”。

## 复现实验配置 / 命令
- v6 timeout-tune profiles（固定 pairs20，使用同一 checkpoint；short/mid/long 各一份）：
  - short：`configs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1.json`
  - mid：`configs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1.json`
  - long：`configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json`

- 最小自检：
  - `conda run -n ros2py310 python infer.py --self-check`

- full（short/mid/long，各 runs=20）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1`

## 代表性运行
- v6 short（full runs=20）：`runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251`
- v6 mid（full runs=20）：`runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902`
- v6 long（full runs=20）：`runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602`

## 结论 / 下一步
- 本版确认：在固定 checkpoint 下，仅调推理期可采纳性筛选参数即可显著降低 short/mid 的 `timeout`（并提升 SR）；long 的 `timeout` 也有小幅下降，但 failure 分布发生迁移（原 timeout pair 部分被修复，但也引入新的 timeout pair）。
- 下一步：在保持 `hybrid/shielded` 口径不变前提下，围绕 `forest_min_progress_m` / `forest_min_od_m` 做小范围 sweep，并额外关注 `argmax_inadmissible_rate` 与 `planning_time_s` 的回退风险。

# 回归复盘：v3p11 → v4p3p1（forest_a，fixed pairs20 v1）

本文用于回答一个核心问题：**“从 v3 到 v4 变差，是算法退化还是口径/采样变化？”**  
结论：两者都有，但需要分清。

---

## 1) 先把口径讲清楚（strict-argmax vs hybrid/shielded）

本仓库推理口径必须与实现一致：

- `strict-argmax`：推理阶段仅 `argmax(Q)`（`--forest-no-fallback`）
- `hybrid/shielded`：允许推理期 replacement/override（`--no-forest-no-fallback`）；当前实现不再启发式 fallback takeover

自 `v3p12` 起，`--forest-no-fallback` 的语义被修正为真正的 strict-argmax。  
因此，历史上“strict no-fallback”的结果若未显式记录语义，可能与 strict-argmax **不可直接横比**。

---

## 2) 固定随机样本（fixed pairs20 v1）

为避免 random pair 漂移，本轮新增并使用固定 pairs：

- short：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
  - source: `runs/repro_20260210_pairgen_forest_a_short20/20260210_221246/table2_kpis_raw.csv`
- long：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
  - source: `runs/repro_20260210_pairgen_forest_a_long20/20260210_221356/table2_kpis_raw.csv`

这些 pairs 通过 baseline-only infer + 可达性筛查（Hybrid A* screening）生成，用于**公平复评测**。

---

## 3) 复评测设置（统一口径）

统一使用 profile `repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300` 作为 infer 参数底座，
并在命令中显式：

- `--no-rand-two-suites`（避免 short/long 自动展开导致“同一 pairs 文件被复用到两个 suite”）
- `--runs 20` + `--rand-pairs-json ...`（固定 pairs）
- `--forest-no-fallback`（strict）或 `--no-forest-no-fallback`（hybrid）
- `--baselines`（置空，不额外跑基线）

对比的两个 checkpoint：

- `v3p11`：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- `v4p3p1`：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/models`

---

## 4) 结果总览（SR + argmax_inad）

### strict-argmax（pure argmax(Q)）

| 版本 | short SR | long SR | argmax_inad(short/long) | 主要失败 |
|---|---:|---:|---|---|
| v3p11 | 0.05 | 0.05 | 0.427 / 0.217 | collision 主导 |
| v4p3p1 | 0.10 | 0.00 | 0.397 / 0.297 | collision 主导 |

**解读**：在 fixed pairs20 上，两版 strict-argmax 都几乎崩溃，差异很小；“strict 下变差”并不是唯一解释。

### hybrid/shielded（允许 replacement/override；历史含 fallback）

| 版本 | short SR | long SR | argmax_inad(short/long) | 主要失败 |
|---|---:|---:|---|---|
| v3p11 | 0.75 | 0.85 | 0.200 / 0.210 | reached 主导，少量 collision/timeout |
| v4p3p1 | 0.55 | 0.70 | 0.351 / 0.292 | timeout 上升，short 仍有 collision |

**解读**：在允许干预的口径下，v4p3p1 相比 v3p11 的确退化（short/long 都降），且 `argmax_inadmissible_rate` 更高，说明策略更常把 Q 最大值落在不可行动作区域，依赖干预才能“救回来”。

### hybrid/shielded（no-heuristic-fallback）

> 2026-02-10 同日补充：移除推理期“启发式 fallback rollout takeover”后重跑，仍使用 fixed pairs20 v1。

| 版本 | short SR | long SR | argmax_inad(short/long) | failure_reason(short/long) |
|---|---:|---:|---|---|
| v3p11 | 0.75 | 0.85 | 0.200 / 0.210 | short: reached15/collision3/timeout2; long: reached17/collision2/timeout1 |
| v4p3p1 | 0.55 | 0.70 | 0.351 / 0.264 | short: reached11/collision2/timeout7; long: reached14/collision1/timeout5 |

**解读**：no-heur 复评测与此前 hybrid 结论一致：`v4p3p1` 仍显著弱于 `v3p11`，主要体现在 timeout 增多和 `argmax_inadmissible_rate` 偏高；本轮四个 run 的 `fallback_rate` 均为 `0.000`。

---

## 5) 结论（回答“为什么越改越差”）

1) **口径因素（重大）**：strict-argmax 与 hybrid 的差异非常大；若版本间口径不一致，会产生“看似大回退/大提升”的错觉。  
2) **真实退化（在 hybrid 下可见）**：即使统一 fixed pairs 与统一命令，v4p3p1 在 hybrid/shielded 下依然弱于 v3p11。

---

## 6) 下一步建议（不改代码前提下）

- 主线研究若要对标论文/门槛，建议以 `strict-argmax` 作为主指标，同时保留 `hybrid/shielded` 作为工程增强指标（必须分开命名与归档）。
- 后续若要提升 strict-argmax，应优先把目标从“调参”切换为“降低 argmax_inadmissible_rate”，否则 strict 口径很难有成功率：
  - 先做小步、单变量的动作空间/奖励/课程或 replay 机制实验；
  - 每轮固定 pairs + strict/hybrid 双口径复评测，避免再被采样漂移误导。

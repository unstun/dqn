# v8p5 设计草案：argmax 不可行时的替换动作排序（replace-ranking）消融

日期：2026-02-23  
目标版本：`v8p5`（patch）

## 0) 背景与问题陈述

当前 V8 迭代基于 `dijkstra8_nocorner`（8 邻接 + 禁止穿角的 Dijkstra 距离场）作为 progress 代价场，在 `SR≈1.0` 的前提下对 `avg_path_length/path_time_s` 有潜在收益（尤其 long）。但在 `v8p3/v8p4` 的 smoke / fixed-pairs 回归中仍观测到：

- `mid collision`（碰撞）
- `long timeout`（超时/不收敛）

在 `forest_no_fallback=false`（shielded/hybrid）口径下，推理与训练都会在 `argmax(Q)` 不可行时触发“替换动作选择”（top-k + mask + fallback）。其中“替换候选的排序规则”会显著影响：

- 是否优先选择更强的“短期进展”（减少绕圈/timeout）
- 是否优先选择更大的“净空”（减少贴障/碰撞）

因此本版先引入一个**纯策略层消融开关**，不改奖励/采样/超参分布，便于隔离变量。

## 1) 设计目标

硬约束：
- 默认行为保持不变（不影响主线已有结果）：新增开关默认采用纯 Q 排序。

次目标（用于 smoke/回归筛查）：
- 在 `configs/pairs_v8p3_smoke_failures.json`（mid collision + long timeout）固定样本上，对比不同替换排序是否能改善 `collision/timeout`。
- 在 `SR≈1.0` 约束下尽量压 `avg_path_length/path_time_s`。

## 2) 方案：新增 `--forest-replace-ranking`

在 `train.py` 与 `infer.py` 均新增 CLI/config 参数：

- `--forest-replace-ranking`（替换动作排序策略）
  - `q`：仅按 Q（可叠加 `--forest-topk-turn-penalty` 的转向惩罚）排序（默认，兼容旧行为）
  - `progress_clearance_q`：优先下一步 `progress_dist`（越小越好），其次 `od`（净空，越大越好），再用 Q 打破平局
  - `clearance_progress_q`：优先 `od`（越大越好），其次 `progress_dist`（越小越好），再用 Q 打破平局

适用范围：仅在 `argmax(Q)` 不可行且允许 fallback（`forest_no_fallback=false`）时生效；strict-argmax 口径下不触发替换逻辑。

## 3) 验证计划（时间优先）

1) 最小自检：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

2) fixed-pairs 回归（先回归再 smoke）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression --forest-replace-ranking clearance_progress_q
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression --forest-replace-ranking q
```

3) 通过回归后再进入 smoke（episodes=150, runs=3）：

```bash
conda run -n ros2py310 python train.py --profile v8p5
conda run -n ros2py310 python infer.py --profile v8p5
```

## 4) 风险与回滚

- 风险：`progress_clearance_q` 可能更激进，若与 mask/horizon 组合不佳，可能在窄通道更容易“贴角钻缝”或产生振荡；`clearance_progress_q` 可能更保守导致路径变长/变慢。
- 回滚：默认 `q` 保持旧行为；即便新模式退化也不会影响主线（只需不在 profile 中启用即可）。


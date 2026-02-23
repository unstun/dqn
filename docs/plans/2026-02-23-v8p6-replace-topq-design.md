# v8p6 设计草案：replace-topq（替换动作候选 Top-Q 约束）

日期：2026-02-23  
目标版本：`v8p6`（patch）

## 0) 背景与问题陈述

在 `v8p5` 中，我们引入了 `--forest-replace-ranking`（替换动作排序策略），用于当 `argmax(Q)` 在 admissible（短视 horizon + clearance + progress）约束下不可行时，如何在 admissible 候选中选择替换动作。

观测到的现象（见 `docs/versions/v8p5/RESULTS.md`）：

- fixed-pairs 回归（mid/long failures）：tie-break（`progress_clearance_q`/`clearance_progress_q`）能修复 `collision/timeout`；
- infer-only smoke（固定 `v7p1` checkpoint，随机 short/mid/long）：tie-break 能明显压 long 的 `avg_path_length/path_time_s`，但 short 出现 `collision=1/3` 回潮；
- 同一随机对上，`q` 能 reached，而 tie-break 会撞，说明 tie-break 在某些状态下**偏离 Q 的幅度过大**，把策略带入了“看起来更接近目标/更宽敞，但长期更危险”的轨迹。

## 1) 设计目标

硬约束：
- **默认行为不变**：新增开关默认不启用（`forest_replace_topq=0`），不影响主线已有结果。
- 在 `SR≈1.0` 的前提下，尽可能压 `avg_path_length/path_time_s`（尤其 long）。

次目标（筛查用）：
- 在 infer-only（固定 `v7p1` checkpoint）随机 short/mid/long 上，验证 tie-break 的 long L/T 改善能否在不牺牲 short SR 的前提下保留。

## 2) 方案：新增 `--forest-replace-topq`

在 `train.py` 与 `infer.py` 均新增参数：

- `--forest-replace-topq`（替换候选 Top-Q 约束）
  - 语义：当进入“替换动作选择”逻辑时，先在 admissible candidates 中按 `Q`（可叠加 `--forest-topk-turn-penalty`）排序取 Top-Q，再对这 Top-Q 做 `--forest-replace-ranking` 的 tie-break（progress/clearance）。
  - `0`：不启用 Top-Q 约束（保持 `v8p5` 的候选集合）
  - `1`：退化为“纯 Q 替换”（即使 `ranking` 设为 tie-break，也只会在 Top-1 内选择，等价于 pure-Q）
  - `>=2`：允许在高 Q 的小集合内用 progress/clearance 选择更高效的动作（预期保留 long L/T 改善，同时降低 short 碰撞风险）

适用范围：仅在 `argmax(Q)` inadmissible 且 `forest_no_fallback=false`（shielded/hybrid）时生效；strict-argmax 不触发替换逻辑。

## 3) 验证计划（时间优先）

1) 单测（本地）：

```bash
conda run -n ros2py310 python -m pytest tests/test_v8p6_replace_topq.py -q
```

2) 最小自检（远端优先）：

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && conda run -n ros2py310 python train.py --self-check"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && conda run -n ros2py310 python infer.py --self-check"
```

3) infer-only smoke（固定 `v7p1` checkpoint；short/mid/long 各 runs=3；保持同一随机对）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 1
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 3
```

通过标准（smoke gate 口径）：
- short/mid/long `success_rate` 均为 `1.0`（至少 short 不允许出现 collision/timeout）
- long `avg_path_length/path_time_s` 相对 `q` 有下降趋势（或至少不显著退化）

## 4) 风险与回滚

- 风险：Top-Q 太小（例如 1）会把 tie-break 的长程效率收益吃掉；Top-Q 太大可能仍会复现 short collision。
- 回滚：默认 `forest_replace_topq=0` 不改变行为；只需不在 profile 中启用即可回退到 `v8p5` 行为。


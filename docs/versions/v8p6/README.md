# v8p6 版本说明（replace-topq：替换动作候选 Top-Q 约束）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p5`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**infer-only smoke 已通过（topq=1/2/3；seed=33；runs=3）；train+infer smoke（episodes=150）NO-GO：short/mid collision=1/3**

## 本版目标

硬约束：
- **默认行为不变**：新增开关默认 `forest_replace_topq=0`，不影响既有主线结果。
- 在 `SR≈1.0` 前提下压 `avg_path_length` / `path_time_s`（尤其 long）。

次目标（筛查用）：
- 在固定 `v7p1` checkpoint 的 infer-only smoke 上，验证 tie-break 的 long L/T 改善能否在不牺牲 short SR 的前提下保留。

## 方法摘要

本版新增 `--forest-replace-topq`（替换动作候选 Top-Q 约束）：

- 触发条件：当 `argmax(Q)` inadmissible 且 `forest_no_fallback=false` 时，系统需要在 admissible candidates 中选择“替换动作”。
- 新逻辑：先按 `Q`（可叠加 `--forest-topk-turn-penalty`）排序取 Top-Q，再在 Top-Q 内应用 `--forest-replace-ranking`（`progress_clearance_q` / `clearance_progress_q`）做 tie-break。
- `forest_replace_topq=0`：不启用约束（兼容 `v8p5` 行为）
- `forest_replace_topq=1`：退化为纯 Q 替换（即使 ranking 设为 tie-break）

直觉：把 tie-break 的“偏离 Q 幅度”控制在高 Q 小集合内，预期可保 long 的 L/T 改善，同时减少 short 的 collision 回潮风险。

## 本轮关键命令（计划执行）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) infer-only smoke（固定 `v7p1` checkpoint；runs=3）

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 1
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 3
```

### 3) train+infer smoke（episodes=150, runs=3）——infer-only 通过后再跑

```bash
conda run -n ros2py310 python train.py --profile v8p6 --forest-replace-topq 3
conda run -n ros2py310 python infer.py --profile v8p6 --forest-replace-topq 3
```

## 代表 run

- infer-only smoke（固定 `v7p1` checkpoint；同一随机对；runs=3）：
  - topq=2（默认）：`runs/v8p6_replace_topq_infer_smoke/20260223_185519`
  - topq=1（≈纯 Q replacement 对照）：`runs/v8p6_replace_topq_infer_smoke/20260223_185553`
  - topq=3（本轮更优候选）：`runs/v8p6_replace_topq_infer_smoke/20260223_185628`
- train+infer smoke（topq=3；episodes=140/150 early-stop；runs=3）：
  - train_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450`
  - infer_run：`runs/v8p6_replace_topq_smoke/train_20260223_191450/infer/20260223_192545`

## 结论（infer-only + train+infer smoke）

- `--forest-replace-topq` 在该随机分布样本上**修复了 v8p5 tie-break 的 short collision 回潮**（short/mid/long 均 `SR=1.0`）。
- 相对 topq=1（≈纯 Q replacement），topq=2/3 能明显压 long 的 `avg_path_length/path_time_s`；其中 topq=3 的三套件均值更优（见 `RESULTS.md`）。
- 但在 train+infer smoke（episodes=150）中，训练产物出现 short/mid `collision=1/3`（NO-GO），且 mid/long L/T 仍落后 baseline；下一步应优先定位 collision 回潮触发态（从 `table2_kpis_raw.csv` 抽失败样本做回归），再决定是否继续训练侧迭代或仅保留推理侧策略。

# v8p5 版本说明（replace-ranking 消融：argmax 不可行时的替换动作排序）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p4`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**回归通过（fixed-pairs PASS；infer-only smoke：tie-break short 有 collision；train+infer smoke 未跑）**

## 本版目标

硬约束（优先）：
- **默认行为不变**：新增开关默认 `forest_replace_ranking=q`，不影响既有主线结果。

次目标（筛查用）：
- 在 `SR≈1.0` 前提下压 `avg_path_length` / `path_time_s`。
- 重点复测 `v8p3` 失败样本对（mid collision + long timeout），看替换排序是否能改善失败类型与路径效率。

## 方法摘要

当 `argmax(Q)` 在短视 `admissible`（horizon + clearance + progress）约束下不可行时，系统会进入“替换动作”逻辑（top-k / mask / fallback）。本版新增 `--forest-replace-ranking`（替换动作排序策略）用于消融：

- `q`：只按 Q（可叠加 `--forest-topk-turn-penalty` 平滑惩罚）排序（默认，兼容旧行为）
- `progress_clearance_q`：优先下一步 `progress_dist`（越小越好），其次 `od`（净空，越大越好），再用 Q 打破平局
- `clearance_progress_q`：优先 `od`（越大越好），其次 `progress_dist`（越小越好），再用 Q 打破平局

说明：该开关只影响“替换候选如何排序”，不改变 admissible 的定义与 mask 生成方式；因此属于**策略层选择偏好**的最小变量消融。

## 本轮关键命令（计划执行）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) 回归（重放 v8p3 smoke failures，infer-only）

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression --forest-replace-ranking clearance_progress_q
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression --forest-replace-ranking q
```

### 3) infer-only smoke（固定 `v7p1` checkpoint，runs=3）——用于筛查泛化风险

```bash
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking progress_clearance_q
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking clearance_progress_q
```

### 4) smoke（episodes=150, runs=3）——infer-only 通过后再跑

```bash
conda run -n ros2py310 python train.py --profile v8p5
conda run -n ros2py310 python infer.py --profile v8p5
```

## 代表 run

- 回归（fixed pairs）：
  - `progress_clearance_q`：`runs/v8p5_replace_ranking_regression/20260222_222704`
  - `progress_clearance_q` + baseline：`runs/v8p5_replace_ranking_regression/20260222_224400`
  - `clearance_progress_q`：`runs/v8p5_replace_ranking_regression/20260222_223308`
  - `q`（基线对照）：`runs/v8p5_replace_ranking_regression/20260222_223339`
- infer-only smoke（固定 `v7p1` checkpoint）：
  - `q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172217`
  - `progress_clearance_q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172252`
  - `clearance_progress_q`：`runs/v8p5_replace_ranking_infer_smoke/20260223_172327`
- train+infer smoke：`N/A`

## 结论（待回填）

- fixed-pairs 回归（mid/long，各 runs=2）消融结果：
  - `q`：FAIL（collision=1/2 + timeout=1/2）
  - `progress_clearance_q`：PASS（`reached=4/4`）
  - `clearance_progress_q`：PASS（`reached=4/4`，但 path/time 更大）
- infer-only smoke（固定 `v7p1` checkpoint，short/mid/long 各 runs=3）筛查结果：
  - `q`：PASS（short/mid/long 均 `SR=1.0`）
  - `progress_clearance_q` / `clearance_progress_q`：FAIL（short 出现 `collision=1/3`，不满足 `SR≈1.0` 的硬约束，但 long 的 L/T 有明显下降趋势）
- 下一步：若继续追求 “`SR≈1.0` 前提下压 L/T”，需要先把 tie-break 策略的 short 安全回潮压住（例如增加额外安全约束/引入更保守的触发条件），再决定是否进入 train+infer smoke gate。

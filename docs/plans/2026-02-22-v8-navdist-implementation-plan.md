# V8 NavDist Progress Distance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变 random start/goal 采样与套件口径的前提下，引入 obstacle-aware 的 progress 距离（grid shortest-path / BFS），用于 reward progress 与推理期 admissible gating，从而在 `SR≈1.0` 前提下压 `avg_path_length/path_time_s`。

**Architecture:** 保留现有欧氏距离场 `_goal_dist_m`（用于采样/课程/套件）；新增独立的 progress 距离场（默认沿用欧氏，`grid4` 时对 `traversable_base` 做 BFS geodesic），并在 reward/gating/fallback 处切换到 progress 距离。

**Tech Stack:** Python 3.10、NumPy、标准库（`collections.deque`）、pytest（仓库已有）。

---

### Task 1: 新增 BFS geodesic 距离函数 + 单测（TDD）

**Files:**
- Modify: `forest_vehicle_dqn/env.py`（新增 `grid4` shortest-path 距离函数）
- Test: `tests/test_forest_progress_distance.py`

**Step 1: 写失败测试**

在 `tests/test_forest_progress_distance.py` 添加：
- 一个小栅格（比如 7×7）带障碍墙，验证：
  - goal cell 距离为 0
  - 墙后不可达区域距离为 `inf`
  - 简单可达点距离为 `k*cell_size_m`（k 为 BFS 步数）

**Step 2: 运行测试确认失败**

Run: `pytest -q tests/test_forest_progress_distance.py -k navdist -vv`  
Expected: FAIL（函数未实现或导入失败）

**Step 3: 实现最小 BFS**

在 `forest_vehicle_dqn/env.py` 实现：
- `grid4_goal_dist_m(traversable: np.ndarray, goal_xy: tuple[int,int], cell_size_m: float) -> np.ndarray`
- 输出 `float32`，不可达为 `inf`

**Step 4: 运行测试确认通过**

Run: `pytest -q tests/test_forest_progress_distance.py -k navdist -vv`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_forest_progress_distance.py forest_vehicle_dqn/env.py
git commit -m "test(env): add grid4 progress-distance BFS"
```

---

### Task 2: 在 `AMRBicycleEnv` 增加 `progress_dist_mode` 并接入 reward progress

**Files:**
- Modify: `forest_vehicle_dqn/env.py`

**Step 1: 写/更新测试（可选增强）**
- 通过一个最小 reset/step（可用 `forest_b` 小图）验证 `progress_dist_mode="grid4"` 时不会崩溃（不要求数值完全一致）。

**Step 2: 实现**
- `AMRBicycleEnv.__init__(..., progress_dist_mode: str = "euclid")`
- 在 `reset(...)` 的最终 goal 确定后调用 `_ensure_progress_dist_field()`（避免在 sampling tries 内重复计算）
- 在 `_step_with_controls(...)` 将 `dist_before/dist_after` 切到 progress 距离
- 保持回退：progress 距离非有限时用欧氏直线距离差

**Step 3: 本地自检**

Run:
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`

**Step 4: Commit**

```bash
git add forest_vehicle_dqn/env.py
git commit -m "feat(env): add navdist progress distance for reward"
```

---

### Task 3: 接入 admissible gating + fallback 动作选择

**Files:**
- Modify: `forest_vehicle_dqn/env.py`

**Step 1: 实现**
- `is_action_admissible(...)` / `admissible_action_mask(...)` / `_fallback_action_short_rollout(...)`
  - 把 `dist0/dist1` 与 rollout 目标距离改为 progress 距离（保持其余安全逻辑不变）

**Step 2: 运行相关单测**

Run: `pytest -q`（或至少跑与 gating 相关的现有测试集）

**Step 3: Commit**

```bash
git add forest_vehicle_dqn/env.py
git commit -m "feat(env): use navdist for admissible gating and fallback"
```

---

### Task 4: CLI/config 透传 `--forest-progress-dist-mode`

**Files:**
- Modify: `forest_vehicle_dqn/cli/train.py`
- Modify: `forest_vehicle_dqn/cli/infer.py`
- (Later) Add: `configs/v8.json`
- (Later) Add: `configs/repro_20260222_v8_navdist_smoke.json`
- (Later) Add: `configs/repro_20260222_v8_navdist_infer_smoke.json`

**Step 1: CLI 增加参数**
- `--forest-progress-dist-mode {euclid,grid4}`，默认 `euclid`
- 创建 `AMRBicycleEnv(...)` 时传入 `progress_dist_mode`

**Step 2: self-check**
- `conda run -n ros2py310 python train.py --self-check`
- `conda run -n ros2py310 python infer.py --self-check`

**Step 3: Commit**

```bash
git add forest_vehicle_dqn/cli/train.py forest_vehicle_dqn/cli/infer.py
git commit -m "feat(cli): add --forest-progress-dist-mode"
```

---

### Task 5: V8 profile + repro config + 版本四件套（占位→随运行更新）

**Files:**
- Add: `configs/v8.json`
- Add: `configs/repro_20260222_v8_navdist_smoke.json`
- Add: `configs/repro_20260222_v8_navdist_infer_smoke.json`
- Add: `docs/versions/v8/README.md`
- Add: `docs/versions/v8/CHANGES.md`
- Add: `docs/versions/v8/RESULTS.md`
- Add: `docs/versions/v8/runs/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/versions/README.md`

**Step 1: 配置**
- `v8.json`：从 `v7p1` 复制，最小改动开启 `forest_progress_dist_mode="grid4"`（其余保持默认/原口径）
- repro configs：写入 smoke 命令（infer-only + train+infer）

**Step 2: 文档占位**
- 先写清楚目标/方法/命令/预期指标口径；结果字段先写 `N/A`，等待实际 run 回填

**Step 3: Commit**

```bash
git add configs/v8.json configs/repro_20260222_v8_navdist_smoke.json configs/repro_20260222_v8_navdist_infer_smoke.json \
  docs/versions/v8/README.md docs/versions/v8/CHANGES.md docs/versions/v8/RESULTS.md docs/versions/v8/runs/README.md \
  README.md README.zh-CN.md docs/versions/README.md
git commit -m "docs(config): add v8 navdist profiles and version docs"
```

---

### Task 6: 远端优先 smoke 运行 + 归档回填

**Files:**
- Modify: `docs/versions/v8/RESULTS.md`
- Modify: `docs/versions/v8/runs/README.md`

**Step 1: 本地 -> 远端同步（不含 runs/）**

Run:
```bash
rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

**Step 2: 远端 self-check**

Run:
```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"
```

**Step 3: infer-only smoke（固定 v7p1 checkpoint，对照）**
- 跑 `euclid` 与 `grid4` 两组（runs=3）对比 KPI

**Step 4: train+infer smoke（episodes=150, runs=3）**

**Step 5: 远端 -> 本地回传 runs/**

**Step 6: 回填 v8 四件套（命令、run_dir、kpi、结论）并 Commit + Push**


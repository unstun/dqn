# v8p3 collision-first fallback 修复 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 short 可复现 `collision`：当 progress mask 为空时，fallback 不应因为 `min_od_m` 过严把所有 collision-free 动作筛空，进而触发 `_fallback_action_short_rollout(...)` 的“even if it still collides”最后兜底。

**Architecture:** 在 `AMRBicycleEnv.admissible_action_mask(...)`（可行动作集合/掩码生成）中，`fallback_to_safe=True` 且 `out` 为空时：
1) 优先回退到 `safe`（`~coll & min_od>=min_od_m`，保持原语义）；
2) 若仍为空但存在 collision-free 动作，则再回退到 `~coll`（放松 `min_od_m`，collision-first）。

**Tech Stack:** Python, numpy, gymnasium, pytest.

---

### Task 1: 写失败单测（复现“safe 被 min_od_m 筛空但仍有 ~coll”）

**Files:**
- Create: `tests/test_v8p3_collision_first_fallback.py`

**Step 1: Write the failing test**

```python
def test_admissible_action_mask_fallback_relaxes_min_od_when_needed() -> None:
    ...
```

测试要点：
- 构造一个小地图（带边界障碍），保证 EDT 计算稳定；
- 选择一个极大 `min_od_m`，使得 `safe=(~coll & min_od>=min_od_m)` 必为空；
- 期待：
  - `fallback_to_safe=False` → mask 为空
  - `fallback_to_safe=True` → mask 非空（至少包含 collision-free 动作）

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_v8p3_collision_first_fallback.py -k admissible_action_mask`
Expected: FAIL（`fallback_to_safe=True` 仍返回空 mask）

**Step 3: Commit**

```bash
git add tests/test_v8p3_collision_first_fallback.py
git commit -m "test(v8p3): reproduce empty safe-mask despite collision-free actions"
```

---

### Task 2: 最小代码修复（collision-first fallback）

**Files:**
- Modify: `forest_vehicle_dqn/env.py`（`AMRBicycleEnv.admissible_action_mask`）

**Step 1: Write minimal implementation**

目标逻辑（伪码）：

```python
if fallback_to_safe and not out.any():
    if safe.any():
        out = safe
    else:
        no_coll = (~coll) & np.isfinite(dist1)
        if no_coll.any():
            out = no_coll
```

**Step 2: Run the new test**

Run: `python -m pytest -q tests/test_v8p3_collision_first_fallback.py -k admissible_action_mask`
Expected: PASS

**Step 3: Run full unit tests**

Run: `python -m pytest -q`
Expected: PASS

**Step 4: Commit**

```bash
git add forest_vehicle_dqn/env.py
git commit -m "fix(v8p3): collision-first fallback when safe mask empty"
```

---

### Task 3: 复现配置 + v8p3 四件套骨架

**Files:**
- Create: `configs/v8p3.json`
- Create: `configs/repro_20260223_v8p3_fallback_safety_smoke.json`
- (Optional) Create: `configs/pairs_v8p2_short_collision_run2.json`
- Create: `docs/versions/v8p3/README.md`
- Create: `docs/versions/v8p3/CHANGES.md`
- Create: `docs/versions/v8p3/RESULTS.md`
- Create: `docs/versions/v8p3/runs/README.md`
- Modify: `configs/INDEX.md`
- Modify: `docs/versions/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Step 1: Add configs**
- `v8p3.json`：基于 `v8p2.json`，仅替换 `out/models` 为 `v8p3_*`，参数保持不动（消融优先）。
- repro：记录最小复现实验命令（建议 `episodes=150, runs=3` 的 smoke 门，以及固定 pair 回归用例）。

**Step 2: Add docs skeleton**
- 四件套按模板填：方法、变更点、命令、结果先写 `N/A`（直到跑完 smoke）。

**Step 3: Minimal self-check**

Run:
```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```
Expected: PASS

**Step 4: Commit**

```bash
git add configs/v8p3.json configs/repro_20260223_v8p3_fallback_safety_smoke.json configs/INDEX.md
git add docs/versions/v8p3 docs/versions/README.md README.md README.zh-CN.md
git commit -m "docs+configs(v8p3): add profile and archive skeleton"
```

---

### Task 4: 远端 smoke（ubuntu-zt，执行后回填 RESULTS）

**Step 1: 同步本地覆盖远端（不含 runs/）**
- 以你现有的同步脚本/rsync 口径为准。

**Step 2: smoke**
- 训练：`conda run -n ros2py310 python train.py --profile v8p3`
- 推理：`conda run -n ros2py310 python infer.py --profile v8p3`

**Step 3: 回传 runs/**
- 将远端本次 `runs/v8p3_*` 回传到本地 `runs/`。

**Step 4: 回填**
- 回填 `docs/versions/v8p3/RESULTS.md` 与 `docs/versions/v8p3/runs/README.md`（真实路径与指标，不得编造）。


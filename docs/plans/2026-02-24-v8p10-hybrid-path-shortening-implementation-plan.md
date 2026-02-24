# v8p10 (hybrid path shortening) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Under the allowed `hybrid/shielded` inference regime (masking/replacement/fallback), reduce RL `avg_path_length/path_time_s` while keeping `success_rate≈1.0`, and ultimately beat the baseline `Hybrid A*-MPC` on fixed pairs20 (gate C). Do **not** change goal tolerances.

**Architecture:** Inference-first ablation. Add a v8p10 profile + reproducible sweep smoke profiles (short/long, fixed pairs3). Primary axis: `forest_progress_dist_mode=dijkstra8_nocorner` with a sweep over `forest_progress_cost_w_clearance`. If still too long, add two minimal replacement ranking modes (`progress_q`, `progress_q_clearance`) shared by both `infer.py` and `train.py`, with unit tests to ensure consistency.

**Tech Stack:** Python 3.10, existing repo train/infer CLIs, JSON configs, pytest, markdown docs.

---

### Task 0: Pre-flight safety (clean + snapshot)

**Files:**
- None

**Step 1: Ensure clean workspace**

Run: `git status -sb`  
Expected: clean.

**Step 2: Snapshot tag**

Run: `git tag -a v8p10-pre-YYYYMMDD -m "pre-v8p10 snapshot" && git push --tags`  
Expected: push succeeds.

---

### Task 1: Add v8p10 profiles + sweep smoke repro configs (pairs3)

**Files:**
- Add: `configs/v8p10.json`
- Add: `configs/repro_YYYYMMDD_v8p10_infer_sweep_short_smoke.json`
- Add: `configs/repro_YYYYMMDD_v8p10_infer_sweep_long_smoke.json`

**Step 1: Validate JSON**

Run: `python -m json.tool <file> >/dev/null` for each added json  
Expected: exit 0.

**Step 2: Commit**

Run:
```bash
git add configs/v8p10.json configs/repro_*v8p10*
git commit -m "v8p10: add infer sweep profiles (progress-dist clearance ablation)"
git push
```

---

### Task 2 (optional): Add replacement ranking modes to reduce detours

**Files:**
- Modify: `forest_vehicle_dqn/cli/infer.py`
- Modify: `forest_vehicle_dqn/cli/train.py`
- Add: `tests/test_v8p10_replace_ranking_progress_q.py`

**Step 1: Add ranking modes**

Add `forest_replace_ranking` modes:
- `progress_q`
- `progress_q_clearance`

Both infer/train must accept the same strings and behave identically for a stub env.

**Step 2: Run unit tests**

Run: `pytest -q tests/test_v8p10_replace_ranking_progress_q.py`  
Expected: PASS.

**Step 3: Commit**

```bash
git add forest_vehicle_dqn/cli/infer.py forest_vehicle_dqn/cli/train.py tests/test_v8p10_replace_ranking_progress_q.py
git commit -m "v8p10: add progress-first replacement ranking modes"
git push
```

---

### Task 3: Add v8p10 version archive skeleton

**Files:**
- Add: `docs/versions/v8p10/README.md`
- Add: `docs/versions/v8p10/CHANGES.md`
- Add: `docs/versions/v8p10/RESULTS.md`
- Add: `docs/versions/v8p10/runs/README.md`
- Modify: `configs/INDEX.md`

**Step 1: Commit**

```bash
git add docs/versions/v8p10 configs/INDEX.md
git commit -m "docs(v8p10): add version archive skeleton"
git push
```

---

### Task 4: Run sweep smoke (short/long, fixed pairs3) with baseline and archive

**Files:**
- Modify: `docs/versions/v8p10/RESULTS.md`
- Modify: `docs/versions/v8p10/runs/README.md`

**Step 1: Sync to remote (preferred)**

```bash
rsync -av --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' ./ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

**Step 2: Run on remote**

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_YYYYMMDD_v8p10_infer_sweep_short_smoke"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_YYYYMMDD_v8p10_infer_sweep_long_smoke"
```

**Step 3: Pull runs back**

```bash
rsync -av ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/<v8p10_out>/ runs/<v8p10_out>/
```

**Step 4: Archive**

Fill `docs/versions/v8p10/RESULTS.md` + `docs/versions/v8p10/runs/README.md` with `run_dir` + KPI paths + baseline comparisons.

**Step 5: Commit**

```bash
git add docs/versions/v8p10/RESULTS.md docs/versions/v8p10/runs/README.md
git commit -m "docs(v8p10): archive sweep smoke results"
git push
```


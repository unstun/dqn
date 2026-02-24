# v8p9 (infer sweep) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible v8p9 inference-sweep workflow (fixed pairs, smoke then full gate C) to search for inference-side settings that keep SR≈1.0 while reducing `avg_path_length` and `path_time_s`.

**Architecture:** No algorithmic code changes. Add `v8p9` profile + two smoke sweep profiles (short/long) pinned to a known model run, plus fixed pairs3 subsets derived from existing pairs20. Add v8p9 version archive skeleton for traceability.

**Tech Stack:** Python 3.10, existing repo train/infer CLIs, JSON configs, markdown docs.

---

### Task 0: Pre-flight safety (clean + snapshot)

**Files:**
- None

**Step 1: Ensure clean workspace**

Run: `git status -sb`  
Expected: clean.

**Step 2: Snapshot tag**

Run: `git tag -a v8p9-pre-YYYYMMDD -m "pre-v8p9 snapshot" && git push --tags`  
Expected: push succeeds.

---

### Task 1: Add v8p9 profiles + fixed pairs3 subsets

**Files:**
- Add: `configs/v8p9.json`
- Add: `configs/repro_YYYYMMDD_v8p9_infer_sweep_short_smoke.json`
- Add: `configs/repro_YYYYMMDD_v8p9_infer_sweep_long_smoke.json`
- Add: `configs/pairs_v8p9_smoke_short3_from_pairs20_*.json`
- Add: `configs/pairs_v8p9_smoke_long3_from_pairs20_*.json`

**Step 1: Validate JSON**

Run: `python -m json.tool <file> >/dev/null` for each added json  
Expected: exit 0.

**Step 2: Commit**

Run:
```bash
git add configs/v8p9.json configs/repro_*v8p9* configs/pairs_v8p9*
git commit -m "v8p9: add infer sweep profiles + pairs3 subsets"
git push
```

---

### Task 2: Add v8p9 archive skeleton

**Files:**
- Add: `docs/versions/v8p9/README.md`
- Add: `docs/versions/v8p9/CHANGES.md`
- Add: `docs/versions/v8p9/RESULTS.md`
- Add: `docs/versions/v8p9/runs/README.md`

**Step 1: Commit**

Run:
```bash
git add docs/versions/v8p9
git commit -m "docs(v8p9): add version archive skeleton"
git push
```

---

### Task 3: Sync to remote and run sweep smoke

**Files:**
- Modify: `docs/versions/v8p9/RESULTS.md`
- Modify: `docs/versions/v8p9/runs/README.md`

**Step 1: Sync code/configs to remote**

Run locally:
```bash
rsync -av --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' ./ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

**Step 2: Run on remote**

Run:
```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_YYYYMMDD_v8p9_infer_sweep_short_smoke"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_YYYYMMDD_v8p9_infer_sweep_long_smoke"
```

**Step 3: Pull runs back**

Run locally (paths depend on `--out`):
```bash
rsync -av ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/<v8p9_out>/ runs/<v8p9_out>/
```

**Step 4: Archive**

Fill `docs/versions/v8p9/RESULTS.md` + `docs/versions/v8p9/runs/README.md` with run_dir + KPI paths.

**Step 5: Commit**

```bash
git add docs/versions/v8p9/RESULTS.md docs/versions/v8p9/runs/README.md
git commit -m "docs(v8p9): archive sweep smoke results"
git push
```

---


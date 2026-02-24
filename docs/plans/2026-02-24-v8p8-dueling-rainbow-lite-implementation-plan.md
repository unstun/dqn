# v8p8 (Dueling + GlobalCNN-Fusion + aux admissibility) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make a DQN-variant agent beat `Hybrid A*-MPC` on the final gate (C): short/long suites with `runs=20` each, with fixed pairs, achieving `SR>=baseline` and strictly lower `avg_path_length` and `path_time_s`.

**Architecture:** Extend existing Q-networks with an optional dueling head; enable stronger global map encoding (globalcnn_fusion + spatial prior); add/enable training-only admissibility auxiliary supervision; keep inference policy in `shielded/hybrid` regime with clearly documented switches; validate via smoke then full20 fixed-pairs gate.

**Tech Stack:** Python 3.10, PyTorch, existing repo training/infer CLIs, pytest.

---

### Task 0: Pre-flight safety (clean + snapshot)

**Files:**
- None

**Step 1: Ensure clean workspace**

Run: `git status -sb`  
Expected: clean (no pending changes).

**Step 2: Pre-change snapshot**

```bash
git add -A
git commit -m "snapshot: pre-v8p8 dueling+globalcnn+aux"
git push
```

Expected: push succeeds; if not, stop and fix remote first.

---

### Task 1: Add dueling head (unit-tested)

**Files:**
- Modify: `forest_vehicle_dqn/networks.py`
- Modify: `forest_vehicle_dqn/agents.py` (pass config knobs through)
- Test: `tests/test_v8p8_dueling_network.py`

**Step 1: Write the failing test**

Create `tests/test_v8p8_dueling_network.py`:
- Instantiate `MLPQNetwork` and `CNNQNetwork` with a new flag (e.g. `dueling=True`) and assert:
  - output shape is `(batch, n_actions)`
  - dueling path is deterministic and finite
  - when `dueling=False`, behavior matches the current single-head path (smoke-level check: same output shape)

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. conda run -n ros2py310 pytest -q tests/test_v8p8_dueling_network.py`  
Expected: FAIL (missing dueling flag / attribute errors).

**Step 3: Implement minimal dueling head**

In `forest_vehicle_dqn/networks.py`:
- Add `dueling: bool = False` and `dueling_hidden_dim` (optional) to network inits.
- Split final head into:
  - value stream `V(s)` -> `(batch, 1)`
  - advantage stream `A(s,a)` -> `(batch, n_actions)`
- Combine as `Q = V + (A - mean(A))`.
- Keep old head as default when `dueling=False`.

In `forest_vehicle_dqn/agents.py`:
- Extend `AgentConfig` with `dueling: bool = False`, `dueling_hidden_dim: int = 256` (or reuse hidden_dim).
- Pass these into network ctor via `_net_kwargs`.

**Step 4: Run tests to verify pass**

Run: `PYTHONPATH=. conda run -n ros2py310 pytest -q tests/test_v8p8_dueling_network.py`  
Expected: PASS.

**Step 5: Commit**

```bash
git add forest_vehicle_dqn/networks.py forest_vehicle_dqn/agents.py tests/test_v8p8_dueling_network.py
git commit -m "v8p8: add optional dueling Q head"
git push
```

---

### Task 2: Wire CLI/config knobs (train + infer)

**Files:**
- Modify: `forest_vehicle_dqn/cli/train.py`
- Modify: `forest_vehicle_dqn/cli/infer.py`
- Add: `configs/v8p8.json`
- Add: `configs/repro_20260224_v8p8_smoke.json`
- Add: `docs/versions/v8p8/README.md`
- Add: `docs/versions/v8p8/CHANGES.md`
- Add: `docs/versions/v8p8/RESULTS.md`
- Add: `docs/versions/v8p8/runs/README.md`
- Modify: `configs/INDEX.md`
- Modify: `docs/versions/README.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

**Step 1: Add CLI args for v8p8 toggles**

In `train.py`:
- Add args: `--dueling` (bool), `--dueling-hidden-dim` (int)
- Ensure they flow into `AgentConfig(...)`.

In `infer.py`:
- No need for dueling flags if loaded from checkpoint, but keep compatibility:
  - Ensure checkpoint load restores correct architecture (store config in checkpoint already).

**Step 2: Add v8p8 profiles**

Create `configs/v8p8.json`:
- Based on `configs/v8p7.json`
- Set:
  - `cnn_backbone=globalcnn_fusion`
  - `cnn_global_spatial_prior=true`
  - `dueling=true`
  - `aux_admissibility_lambda` > 0 (start small, e.g. 0.05)
  - Keep goal tolerances unchanged
  - Keep `forest_goal_approach_override=true` but tune later

Create `configs/repro_20260224_v8p8_smoke.json`:
- episodes=150, runs=3, seed fixed
- output directory `v8p8_smoke`

**Step 3: Update docs + indices**

- Add `docs/versions/v8p8/` four-pack, with placeholders for results.
- Update `configs/INDEX.md`, `docs/versions/README.md`, root READMEs to point active V8 iteration to v8p8 once smoke exists.

**Step 4: Commit**

Commit config/docs wiring separately from algorithmic changes if possible.

---

### Task 3: Smoke on remote (ubuntu-zt) and archive results

**Files:**
- Modify: `docs/versions/v8p8/RESULTS.md`
- Modify: `docs/versions/v8p8/runs/README.md`

**Step 1: Sync to remote (exclude runs/)**

Run locally:
```bash
rsync -av --delete --exclude runs/ /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

**Step 2: Run smoke on remote**

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v8p8"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v8p8"
```

Expected: produces `runs/v8p8_*/train_YYYY.../infer/.../table2_kpis_mean_raw.csv`.

**Step 3: Pull runs back**

```bash
rsync -av ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/<v8p8_out>/<train_ts>/ /home/sun/phdproject/dqn/dqn/runs/<v8p8_out>/<train_ts>/
```

**Step 4: Archive**

Update v8p8 RESULTS + runs README with:
- run_dir, run_json, kpi paths
- baseline comparison
- whether it is a GO for full20

**Step 5: Commit**

```bash
git add docs/versions/v8p8/RESULTS.md docs/versions/v8p8/runs/README.md
git commit -m "docs(v8p8): archive smoke results"
git push
```

---

### Task 4: Full gate C (short/long pairs20, runs=20)

**Files:**
- Modify: `docs/versions/v8p8/RESULTS.md`
- Modify: `docs/versions/v8p8/runs/README.md`

**Step 1: Run full20 (short) on remote**

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p8_full20_pairs_short"
```

**Step 2: Run full20 (long) on remote**

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p8_full20_pairs_long"
```

**Step 3: Pull runs back and compute verdict**

Pull the run dirs back, then read `table2_kpis_mean_raw.csv`:
- Check C: SR>=baseline and L/T strictly smaller on both short and long.

**Step 4: Archive + commit**

```bash
git add docs/versions/v8p8/RESULTS.md docs/versions/v8p8/runs/README.md
git commit -m "docs(v8p8): archive full20 fixed-pairs gate"
git push
```

---

### Task 5: Parameter sweep (only if C not met)

**Files:**
- Modify: `configs/v8p8.json`
- Add: `configs/repro_20260224_v8p8_full20_sweep_*.json`
- Modify: `docs/versions/v8p8/CHANGES.md` / `RESULTS.md`

**Knobs to sweep (small grid):**
- `forest_goal_approach_speed_factor` in `{0.8, 0.9, 1.0}`
- `forest_goal_approach_dist_m` in `{1.5, 2.5}`
- `aux_admissibility_lambda` in `{0.0, 0.02, 0.05}`

**Rule:** change one group at a time; every run archived with run_dir + kpi paths; no cherry-picking.


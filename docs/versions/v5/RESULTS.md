# v5 - 结果

## 代表性正式指标（fixed pairs20 v1）

### strict-argmax（pure argmax，四基线同场）
- short KPI：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/table2_kpis_mean_raw.csv`
  - CNN：SR=`0.05`，avg_path_length=`12.1`，path_time_s=`7.35`，argmax_inad=`0.427`，inference_time_s=`0.06606`
  - Hybrid A*：SR=`1.00`，avg_path_length=`15.1907`，path_time_s=`7.5954`，inference_time_s=`0.979`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`15.5982`，path_time_s=`9.2342`，inference_time_s=`1.40354`
  - RRT*：SR=`0.95`，avg_path_length=`14.8333`，path_time_s=`7.4166`，inference_time_s=`0.491`
  - RRT-MPC：SR=`0.95`，avg_path_length=`15.3188`，path_time_s=`8.9711`，inference_time_s=`0.85829`
- long KPI：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/table2_kpis_mean_raw.csv`
  - CNN：SR=`0.05`，avg_path_length=`31.432`，path_time_s=`17.4`，argmax_inad=`0.217`，inference_time_s=`0.08484`
  - Hybrid A*：SR=`1.00`，avg_path_length=`32.4054`，path_time_s=`16.2027`，inference_time_s=`5.35934`
  - Hybrid A*-MPC：SR=`0.90`，avg_path_length=`32.8275`，path_time_s=`17.7861`，inference_time_s=`6.57689`
  - RRT*：SR=`1.00`，avg_path_length=`33.5975`，path_time_s=`16.7988`，inference_time_s=`0.99963`
  - RRT-MPC：SR=`0.95`，avg_path_length=`32.6042`，path_time_s=`17.7447`，inference_time_s=`1.54921`

### hybrid/shielded（allow replacement/override，四基线同场）
- short KPI：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/table2_kpis_mean_raw.csv`
  - CNN：SR=`0.75`，avg_path_length=`15.3238`，path_time_s=`9.49`，argmax_inad=`0.200`，inference_time_s=`0.4327`
  - Hybrid A*：SR=`1.00`，avg_path_length=`15.1907`，path_time_s=`7.5954`，inference_time_s=`0.88337`
  - Hybrid A*-MPC：SR=`0.95`，avg_path_length=`15.5982`，path_time_s=`9.2342`，inference_time_s=`1.23579`
  - RRT*：SR=`0.95`，avg_path_length=`14.8333`，path_time_s=`7.4166`，inference_time_s=`0.46439`
  - RRT-MPC：SR=`0.95`，avg_path_length=`15.3188`，path_time_s=`8.9711`，inference_time_s=`0.72222`
- long KPI：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/table2_kpis_mean_raw.csv`
  - CNN：SR=`0.85`，avg_path_length=`33.4719`，path_time_s=`19.8147`，argmax_inad=`0.210`，inference_time_s=`0.46675`
  - Hybrid A*：SR=`1.00`，avg_path_length=`32.4054`，path_time_s=`16.2027`，inference_time_s=`4.9839`
  - Hybrid A*-MPC：SR=`0.90`，avg_path_length=`32.8275`，path_time_s=`17.7861`，inference_time_s=`5.61012`
  - RRT*：SR=`1.00`，avg_path_length=`33.5975`，path_time_s=`16.7988`，inference_time_s=`0.8352`
  - RRT-MPC：SR=`0.95`，avg_path_length=`32.6042`，path_time_s=`17.7447`，inference_time_s=`1.32437`

## 门槛检查（按主口径 hybrid/shielded，对标 Hybrid A*-MPC）

### short
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.75 >= 0.95` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `15.3238 < 15.5982` → **PASS**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `9.49 < 9.2342` → **FAIL**

### long
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`: `0.85 >= 0.90` → **FAIL**
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`: `33.4719 < 32.8275` → **FAIL**
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`: `19.8147 < 17.7861` → **FAIL**

### 最终门槛状态
- `未通过`

## 失败分析（failure_reason 分布，四基线同场）

### strict-argmax
- short raw：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/table2_kpis_raw.csv`
  - CNN：`collision=19, reached=1`
  - Hybrid A*：`reached=20`
  - Hybrid A*-MPC：`reached=19, collision=1`
  - RRT*：`reached=19, planner_fail=1`
  - RRT-MPC：`reached=19, planner_fail=1`
- long raw：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/table2_kpis_raw.csv`
  - CNN：`collision=19, reached=1`
  - Hybrid A*：`reached=20`
  - Hybrid A*-MPC：`reached=18, collision=1, truncated=1`
  - RRT*：`reached=20`
  - RRT-MPC：`reached=19, collision=1`

### hybrid/shielded
- short raw：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/table2_kpis_raw.csv`
  - CNN：`reached=15, collision=3, timeout=2`
  - Hybrid A*：`reached=20`
  - Hybrid A*-MPC：`reached=19, collision=1`
  - RRT*：`reached=19, planner_fail=1`
  - RRT-MPC：`reached=19, planner_fail=1`
- long raw：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/table2_kpis_raw.csv`
  - CNN：`reached=17, collision=2, timeout=1`
  - Hybrid A*：`reached=20`
  - Hybrid A*-MPC：`reached=18, collision=1, truncated=1`
  - RRT*：`reached=20`
  - RRT-MPC：`reached=19, collision=1`

## 运行过程记录（2026-02-11）

### self-check
- command:
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（`device_ok=cuda:0`）

### retrain（v3p11 smoke 复现）
- train command:
  - `conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke --out repro_20260211_v5_retrain_v3p11_smoke`
- train run_dir：`runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356`
- train run_json：`runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356/configs/run.json`
- train_eval：`runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356/training_eval.csv`

- infer smoke command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke --models runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356/models --out repro_20260211_v5_retrain_v3p11_smoke_infer10`
- infer run_dir：`runs/repro_20260211_v5_retrain_v3p11_smoke_infer10/20260211_081345`
- infer kpi：`runs/repro_20260211_v5_retrain_v3p11_smoke_infer10/20260211_081345/table2_kpis_mean_raw.csv`

### 四基线同场对照（fixed pairs20）
- strict short command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_short_pairs20_v1`
- strict short run_dir：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`
- strict short run_json：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/configs/run.json`
- strict short kpi：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/table2_kpis_mean_raw.csv`

- strict long command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_long_pairs20_v1`
- strict long run_dir：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`
- strict long run_json：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/configs/run.json`
- strict long kpi：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/table2_kpis_mean_raw.csv`

- hybrid short command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_short_pairs20_v1`
- hybrid short run_dir：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`
- hybrid short run_json：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/configs/run.json`
- hybrid short kpi：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/table2_kpis_mean_raw.csv`

- hybrid long command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_long_pairs20_v1`
- hybrid long run_dir：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`
- hybrid long run_json：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/configs/run.json`
- hybrid long kpi：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/table2_kpis_mean_raw.csv`

### max_steps=2400（120s）诊断复评（hybrid long，固定 pairs20）
- command:
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --baselines hybrid_astar_mpc --out repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1 --max-steps 2400`
- run_dir：`runs/repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1/20260211_140420`
- run_json：`runs/repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1/20260211_140420/configs/run.json`
- kpi：`runs/repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1/20260211_140420/table2_kpis_mean_raw.csv`
- 结果要点：
  - CNN：SR=`0.85`（未提升），failures=`reached=17, collision=2, timeout=1`
  - timeout case：`run_idx=19` 的 `path_time_s=120.0`（依然跑满上限仍未到达）
  - Hybrid A*-MPC：SR=`0.90`，failures=`reached=18, collision=1, truncated=1`

## 结论
- `v5` 当前主口径（hybrid/shielded）在同场四基线对照下，仍优于 strict，但 SR 仍低于 `Hybrid A*` / `Hybrid A*-MPC` / `RRT*` / `RRT-MPC`。
- 按最终门槛（对标 `Hybrid A*-MPC` 的三条不等式）仍未通过。

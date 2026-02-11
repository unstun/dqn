# v4p3p1 - 运行记录

## 代表性运行（当前）
- train run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958`
- train run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/configs/run.json`
- infer run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044`
- infer run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044/table2_kpis_mean_raw.csv`

## 本版本已知运行
- `conda run -n ros2py310 python train.py --self-check`（通过）
- `conda run -n ros2py310 python infer.py --self-check`（通过）
- `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958`（完成）
- `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044`（完成）

## 2026-02-10 追加：fixed pairs20 v1 复评测（strict / hybrid）
- strict-argmax（short）：`runs/repro_20260210_reval_v4p3p1_strict_short_pairs20_v1/20260210_222553`
- strict-argmax（long）：`runs/repro_20260210_reval_v4p3p1_strict_long_pairs20_v1/20260210_222618`
- hybrid/shielded（short）：`runs/repro_20260210_reval_v4p3p1_hybrid_short_pairs20_v1/20260210_222643`
- hybrid/shielded（long）：`runs/repro_20260210_reval_v4p3p1_hybrid_long_pairs20_v1/20260210_222815`

## 2026-02-10 追加：fixed pairs20 v1 复评测（hybrid no-heuristic-fallback）
- hybrid/shielded no-heur（short）：`runs/repro_20260210_reval_v4p3p1_hybrid_short_pairs20_noheur_v1/20260210_231320`
- hybrid/shielded no-heur（long）：`runs/repro_20260210_reval_v4p3p1_hybrid_long_pairs20_noheur_v1/20260210_231433`

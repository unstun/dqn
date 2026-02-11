# v5 - 运行记录

## 代表性运行（最新主结果：四基线同场）
- strict short run_dir: `runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`
- strict short run_json: `runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/configs/run.json`
- strict short kpi: `runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/table2_kpis_mean_raw.csv`

- strict long run_dir: `runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`
- strict long run_json: `runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/configs/run.json`
- strict long kpi: `runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/table2_kpis_mean_raw.csv`

- hybrid short run_dir: `runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`
- hybrid short run_json: `runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/configs/run.json`
- hybrid short kpi: `runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/table2_kpis_mean_raw.csv`

- hybrid long run_dir: `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`
- hybrid long run_json: `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/configs/run.json`
- hybrid long kpi: `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/table2_kpis_mean_raw.csv`

## mid 套件运行（14–42m）
- pairgen run_dir: `runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822`
- pairgen run_json: `runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822/configs/run.json`
- pairgen kpi: `runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822/table2_kpis_mean_raw.csv`

- strict mid smoke run_dir: `runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838`
- strict mid smoke run_json: `runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838/configs/run.json`
- strict mid smoke kpi: `runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838/table2_kpis_mean_raw.csv`

- hybrid mid smoke run_dir: `runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851`
- hybrid mid smoke run_json: `runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851/configs/run.json`
- hybrid mid smoke kpi: `runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851/table2_kpis_mean_raw.csv`

## train300 + infer20 运行（short/mid/long）
- train run_dir: `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840`
- train run_json: `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/configs/run.json`
- train eval: `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/training_eval.csv`
- train meta: `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/configs/train_meta_forest_a.json`

- infer short run_dir: `runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605`
- infer short run_json: `runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605/configs/run.json`
- infer short kpi: `runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605/table2_kpis_mean_raw.csv`
- infer short raw: `runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605/table2_kpis_raw.csv`

- infer mid run_dir: `runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304`
- infer mid run_json: `runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304/configs/run.json`
- infer mid kpi: `runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304/table2_kpis_mean_raw.csv`
- infer mid raw: `runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304/table2_kpis_raw.csv`

- infer long run_dir: `runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706`
- infer long run_json: `runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706/configs/run.json`
- infer long kpi: `runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706/table2_kpis_mean_raw.csv`
- infer long raw: `runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706/table2_kpis_raw.csv`

## 主模型来源
- checkpoint models: `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- 说明：v5 主结果固定复用该历史 checkpoint，避免“重训随机性”影响版本口径。

## 本版本已知运行
- `conda run -n ros2py310 python train.py --self-check`（通过）
- `conda run -n ros2py310 python infer.py --self-check`（通过）
- `runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356`（重训复现）
- `runs/repro_20260211_v5_retrain_v3p11_smoke_infer10/20260211_081345`（重训 smoke infer）
- `runs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1/20260211_081452`
- `runs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1/20260211_081507`
- `runs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1/20260211_081529`
- `runs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1/20260211_081548`
- `runs/repro_20260211_v5_baseline_hybrid_short_pairs20_v1/20260211_090612`（早期 baseline-only）
- `runs/repro_20260211_v5_baseline_hybrid_long_pairs20_v1/20260211_090655`（早期 baseline-only）
- `runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`（四基线同场）
- `runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`（四基线同场）
- `runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`（四基线同场）
- `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`（四基线同场）
- `runs/repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1/20260211_140420`（max_steps=2400≈120s 诊断复评）
- `runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822`（mid pairgen）
- `runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838`（mid strict smoke）
- `runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851`（mid hybrid smoke）
- `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840`（train300，early-stop at 150）
- `runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605`（train300 后 short infer20）
- `runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304`（train300 后 mid infer20）
- `runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706`（train300 后 long infer20）

## 基线对照路径（fixed pairs20，四基线同场）
- strict short kpi: `runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538/table2_kpis_mean_raw.csv`
- strict long kpi: `runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712/table2_kpis_mean_raw.csv`
- hybrid short kpi: `runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220/table2_kpis_mean_raw.csv`
- hybrid long kpi: `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351/table2_kpis_mean_raw.csv`

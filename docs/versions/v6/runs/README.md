# v6 - 运行记录

## 代表性运行（full runs=20，fixed pairs20）
- v6 short run_dir: `runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251`
- v6 short run_json: `runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251/configs/run.json`
- v6 short kpi: `runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251/table2_kpis_mean_raw.csv`

- v6 mid run_dir: `runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902`
- v6 mid run_json: `runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902/configs/run.json`
- v6 mid kpi: `runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902/table2_kpis_mean_raw.csv`

- v6 long run_dir: `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602`
- v6 long run_json: `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602/configs/run.json`
- v6 long kpi: `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602/table2_kpis_mean_raw.csv`

## smoke（long，runs=5）
- v6 tuned long smoke run_dir: `runs/repro_20260211_v6_smoke_timeout_tune_hybrid_long_pairs5_v1/20260211_213931`
- v6 tuned long smoke run_json: `runs/repro_20260211_v6_smoke_timeout_tune_hybrid_long_pairs5_v1/20260211_213931/configs/run.json`
- v6 tuned long smoke kpi: `runs/repro_20260211_v6_smoke_timeout_tune_hybrid_long_pairs5_v1/20260211_213931/table2_kpis_mean_raw.csv`

- baseline（v5 params）long smoke run_dir: `runs/repro_20260211_v6_smoke_baseline_v5params_hybrid_long_pairs5_v1/20260211_214035`
- baseline（v5 params）long smoke run_json: `runs/repro_20260211_v6_smoke_baseline_v5params_hybrid_long_pairs5_v1/20260211_214035/configs/run.json`
- baseline（v5 params）long smoke kpi: `runs/repro_20260211_v6_smoke_baseline_v5params_hybrid_long_pairs5_v1/20260211_214035/table2_kpis_mean_raw.csv`

## 本版本已知运行
- `conda run -n ros2py310 python train.py --self-check`（通过）
- `conda run -n ros2py310 python infer.py --self-check`（通过）
- `runs/repro_20260211_v6_smoke_timeout_tune_hybrid_long_pairs5_v1/20260211_213931`（long smoke）
- `runs/repro_20260211_v6_smoke_baseline_v5params_hybrid_long_pairs5_v1/20260211_214035`（long smoke baseline）
- `runs/repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1/20260211_215251`（short full20）
- `runs/repro_20260211_v6_timeout_tune_hybrid_mid_pairs20_v1/20260211_214902`（mid full20）
- `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602`（long full20）

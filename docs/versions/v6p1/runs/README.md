# v6p1 - 运行记录

> 说明：本文件用于列出本版本相关的已执行 run（含 self-check/smoke/full）。每条至少给出 run_dir、run_json、kpi 路径；如开启 traces，补充 traces 路径。

## 自检
- `conda run -n ros2py310 python infer.py --self-check`（通过）

## long union10 smoke sweep（均开启 traces）
- baseline（v6 params）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_v6params_v1/20260211_235257`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_v6params_v1/20260211_235257/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_v6params_v1/20260211_235257/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_v6params_v1/20260211_235257/traces/`
- varA（min_od=0.0）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA_minod0_v1/20260211_235932`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA_minod0_v1/20260211_235932/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA_minod0_v1/20260211_235932/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA_minod0_v1/20260211_235932/traces/`
- varB（min_progress=-0.02）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varB_minprog_neg002_v1/20260212_000153`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varB_minprog_neg002_v1/20260212_000153/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varB_minprog_neg002_v1/20260212_000153/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varB_minprog_neg002_v1/20260212_000153/traces/`
- varA2（min_od=0.01）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA2_minod001_v1/20260212_000630`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA2_minod001_v1/20260212_000630/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA2_minod001_v1/20260212_000630/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varA2_minod001_v1/20260212_000630/traces/`
- varC（adm_horizon=20）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varC_h20_v1/20260212_000843`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varC_h20_v1/20260212_000843/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varC_h20_v1/20260212_000843/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varC_h20_v1/20260212_000843/traces/`
- varD（min_od=0.0 + adm_horizon=40）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD_minod0_h40_v1/20260212_001049`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD_minod0_h40_v1/20260212_001049/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD_minod0_h40_v1/20260212_001049/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD_minod0_h40_v1/20260212_001049/traces/`
- varD1（min_od=0.0 + adm_horizon=35）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD1_minod0_h35_v1/20260212_001447`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD1_minod0_h35_v1/20260212_001447/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD1_minod0_h35_v1/20260212_001447/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD1_minod0_h35_v1/20260212_001447/traces/`
- varD0（min_od=0.0 + adm_horizon=34）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD0_minod0_h34_v1/20260212_002139`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD0_minod0_h34_v1/20260212_002139/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD0_minod0_h34_v1/20260212_002139/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varD0_minod0_h34_v1/20260212_002139/traces/`
- varE（min_od=0.005 + adm_horizon=35）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE_minod005_h35_v1/20260212_002701`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE_minod005_h35_v1/20260212_002701/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE_minod005_h35_v1/20260212_002701/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE_minod005_h35_v1/20260212_002701/traces/`
- varE1（最终选中；min_od=0.002 + adm_horizon=35）：
  - run_dir：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225`
  - run_json：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225/table2_kpis_mean_raw.csv`
  - traces：`runs/repro_20260211_v6p1_smoke_long_timeout_union10_varE1_minod002_h35_v1/20260212_003225/traces/`

## full runs=20（fixed pairs20）
- long（v6p1 最终采用）：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414`
  - run_json：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414/table2_kpis_mean_raw.csv`
- mid（回归检查）：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1/20260212_003901`
  - run_json：`runs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1/20260212_003901/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1/20260212_003901/table2_kpis_mean_raw.csv`
- short（迁移失败示例；不建议采用）：
  - run_dir：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744`
  - run_json：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/configs/run.json`
  - kpi：`runs/repro_20260211_v6p1_timeout_tune_hybrid_short_pairs20_v1/20260212_003744/table2_kpis_mean_raw.csv`


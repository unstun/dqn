# v8p9 runs 记录（可追溯路径）

## 1) 推理侧 sweep smoke（fixed pairs3，runs=3）

固定 pairs3（从 pairs20 子集抽取）：
- short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
- long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`

2026-02-24（runs=3）：
- 默认候选（profile 默认参数）：
  - short：`runs/v8p9_infer_sweep_short_pairs3_smoke/20260224_114743`
  - long：`runs/v8p9_infer_sweep_long_pairs3_smoke/20260224_114743`
- varA（v8p8-like：`min_prog=0.01, min_od=0.02, topq=3, tp=0.0, sf=0.9`）：
  - short：`runs/v8p9_sweep_short_varA_v8p8like/20260224_114851`
  - long：`runs/v8p9_sweep_long_varA_v8p8like/20260224_114852`
- varB（od=0：`min_prog=0.01, min_od=0.0, topq=3, tp=0.0, sf=0.9`）：
  - short：`runs/v8p9_sweep_short_varB_od0/20260224_115005`
  - long：`runs/v8p9_sweep_long_varB_od0/20260224_115006`
- varC（q-rank：`ranking=q, topq=0`；SR 退化）：
  - short：`runs/v8p9_sweep_short_varC_qrank/20260224_115110`
  - long：`runs/v8p9_sweep_long_varC_qrank/20260224_115110`
- varD（`min_prog=0.02, min_od=0.0, topq=3, tp=0.0, sf=0.9`）：
  - short：`runs/v8p9_sweep_short_varD_mp0p02_od0/20260224_115140`
  - long：`runs/v8p9_sweep_long_varD_mp0p02_od0/20260224_115141`

## 2) full gate（C：fixed pairs20）

- short run_dir：`N/A`
- long run_dir：`N/A`

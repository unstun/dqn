# v8p8 runs 记录（可追溯路径）

## 1) smoke（episodes=150, runs=3）

- profile：`configs/repro_20260224_v8p8_smoke.json`
- train_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059`
- infer_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556`
- kpi_mean_raw：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_mean_raw.csv`
- kpi_raw：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_raw.csv`

## 2) 推理侧消融（固定 pairs3，runs=3）

- short pairs：`configs/pairs_v8p8_smoke_short3_20260224_110556.json`
- long pairs：`configs/pairs_v8p8_smoke_long3_20260224_110556.json`
- short r=1.5：`runs/v8p8_ablate_pairs3_short_r15_sf0p9/20260224_111755`
- short r=2.5：`runs/v8p8_ablate_pairs3_short_r25_sf0p9/20260224_111838`
- long r=1.5：`runs/v8p8_ablate_pairs3_long_r15_sf0p9/20260224_111905`
- long r=2.5：`runs/v8p8_ablate_pairs3_long_r25_sf0p9/20260224_111932`

## 3) full gate（C：fixed pairs20）

- short run_dir：`N/A`
- long run_dir：`N/A`

# v8p16 — runs 索引（待补）

> 说明：本目录只做“路径索引”，不复制大文件。每次运行结束必须更新本文件，保证可追溯。

## 1) 训练

- train smoke（episodes=150）
  - config：`configs/repro_20260224_v8p16_train_smoke.json`
  - run_dir：`runs/v8p16/train_20260224_020118`
  - models_dir：`runs/v8p16/train_20260224_020118/models`
  - checkpoint：`runs/v8p16/train_20260224_020118/models/forest_a/cnn-ddqn.pt`

## 2) 推理（fixed pairs3 + baseline 同跑）

- infer smoke short（runs=3）
  - config：`configs/repro_20260224_v8p16_infer_smoke_short.json`
  - run_dir：`runs/v8p16_infer_smoke_short_pairs3/20260224_023632`
  - kpi：`runs/v8p16_infer_smoke_short_pairs3/20260224_023632/table2_kpis_mean_raw.csv`
- infer smoke long（runs=3）
  - config：`configs/repro_20260224_v8p16_infer_smoke_long.json`
  - run_dir：`runs/v8p16_infer_smoke_long_pairs3/20260224_023542`
  - kpi：`runs/v8p16_infer_smoke_long_pairs3/20260224_023542/table2_kpis_mean_raw.csv`

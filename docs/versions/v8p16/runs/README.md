# v8p16 — runs 索引（待补）

> 说明：本目录只做“路径索引”，不复制大文件。每次运行结束必须更新本文件，保证可追溯。

## 1) 训练

- train smoke（episodes=150）
  - config：`configs/repro_20260224_v8p16_train_smoke.json`
  - run_dir：N/A
  - models_dir：N/A

## 2) 推理（fixed pairs3 + baseline 同跑）

- infer smoke short（runs=3）
  - config：`configs/repro_20260224_v8p16_infer_smoke_short.json`
  - run_dir：N/A
  - kpi：N/A
- infer smoke long（runs=3）
  - config：`configs/repro_20260224_v8p16_infer_smoke_long.json`
  - run_dir：N/A
  - kpi：N/A


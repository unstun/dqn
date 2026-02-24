# v8p15 变更清单（相对 v8p14）

## 1) 配置改动

- 新增版本入口（infer-only：加载 `v8p11` 权重）：
  - `configs/v8p15.json`
- 新增 long 推理 sweep 可复现配置（fixed pairs3，baseline 同跑；固定 `w=1.5, mp=-0.05`）：
  - `configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p2.json`
  - `configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p3.json`
  - `configs/repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p5.json`

## 2) 文档归档

- 新增版本四件套目录：
  - `docs/versions/v8p15/README.md`
  - `docs/versions/v8p15/CHANGES.md`
  - `docs/versions/v8p15/RESULTS.md`
  - `docs/versions/v8p15/runs/README.md`


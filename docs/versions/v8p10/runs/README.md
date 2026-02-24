# v8p10 runs 记录（可追溯路径）

## 1) 推理侧 sweep smoke（fixed pairs3，runs=3）

固定 pairs3（从 pairs20 子集抽取）：
- short pairs：`configs/pairs_v8p9_smoke_short3_from_pairs20_v1_20260224.json`
- long pairs：`configs/pairs_v8p9_smoke_long3_from_pairs20_v1_20260224.json`

2026-02-24（runs=3，ranking=progress_q；扫 `w_clearance`）：
- short：
  - w=0.0：`runs/v8p10_infer_sweep_short_pairs3_smoke/20260224_134521`
  - w=0.5：`runs/v8p10_sweep_short_w0p5/20260224_135059`
  - w=1.0：`runs/v8p10_sweep_short_w1p0/20260224_135116`
  - w=2.0：`runs/v8p10_sweep_short_w2p0/20260224_135134`（当前 best）
- long：
  - w=0.0：`runs/v8p10_infer_sweep_long_pairs3_smoke/20260224_134539`
  - w=0.5：`runs/v8p10_sweep_long_w0p5/20260224_134940`
  - w=1.0：`runs/v8p10_sweep_long_w1p0/20260224_135010`
  - w=2.0：`runs/v8p10_sweep_long_w2p0/20260224_135035`（当前 best）

## 2) full gate（C：fixed pairs20）

- short run_dir：`N/A`
- long run_dir：`N/A`

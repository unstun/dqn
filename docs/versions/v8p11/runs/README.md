# v8p11 runs 记录（可追溯路径）

## 1) train smoke（episodes=150）

- train run_dir（成功）：`runs/v8p11/train_20260224_151042`
  - flow log：`runs/v8p11/train_20260224_151042/train_flow.log`
  - 模型：`runs/v8p11/train_20260224_151042/models/forest_a/cnn-ddqn.pt`
  - 训练过程评测：`runs/v8p11/train_20260224_151042/training_eval.csv`

- 失败 run（已归档，便于追溯）：
  - `runs/v8p11/train_20260224_142154`（旧代码校验漏更新：`forest_replace_ranking=progress_q` 被拒绝，训练在 `Algo start` 后报错退出）
  - `runs/v8p11/train_20260224_150412`（同上）

## 2) infer smoke（fixed pairs3，runs=3）

- short run_dir：`runs/v8p11_infer_smoke_short_pairs3/20260224_152858`
  - kpi mean raw：`runs/v8p11_infer_smoke_short_pairs3/20260224_152858/table2_kpis_mean_raw.csv`
- long run_dir：`runs/v8p11_infer_smoke_long_pairs3/20260224_152917`
  - kpi mean raw：`runs/v8p11_infer_smoke_long_pairs3/20260224_152917/table2_kpis_mean_raw.csv`

## 2.1) long 推理 sweep：`forest_progress_cost_w_clearance`（fixed pairs3，runs=3）

- `w=0.0`：`runs/v8p11_sweep_long_pairs3_w0p0/20260224_155551`
- `w=0.5`：`runs/v8p11_sweep_long_pairs3_w0p5/20260224_155610`
- `w=1.0`：`runs/v8p11_sweep_long_pairs3_w1p0/20260224_155624`
- `w=1.5`：`runs/v8p11_sweep_long_pairs3_w1p5/20260224_155639`
- `w=2.0`：`runs/v8p11_sweep_long_pairs3_w2p0/20260224_155655`
- `w=2.5`：`runs/v8p11_sweep_long_pairs3_w2p5/20260224_155709`

## 3) full gate（C：fixed pairs20）

- short run_dir：`N/A`
- long run_dir：`N/A`

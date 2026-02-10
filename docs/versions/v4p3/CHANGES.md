# v4p3 代码/配置改动

## 相对 v4p2 的核心改动
- 新增配置：`configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json`
  - `episodes: 300`（满足“至少 300 轮训练后再评估”）。
  - `aux_admissibility_lambda: 0.01`（固定 `v4p2 iter3` 的 aux 强度）。
  - `train.out/infer.out` 切换为 `v4p3_smoke300` 命名。
  - `infer.runs: 10`（smoke 口径）。
- 本轮无算法代码改动，主要是配置与实验归档更新。

## 运行归档更新（2026-02-10）
- 新增 `self-check -> train(300 episodes) -> infer(runs=10)` 完整链路记录。
- 新增版本目录四件套：
  - `docs/versions/v4p3/README.md`
  - `docs/versions/v4p3/CHANGES.md`
  - `docs/versions/v4p3/RESULTS.md`
  - `docs/versions/v4p3/runs/README.md`

## 受影响文件清单
- `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json`
- `docs/versions/v4p3/README.md`
- `docs/versions/v4p3/CHANGES.md`
- `docs/versions/v4p3/RESULTS.md`
- `docs/versions/v4p3/runs/README.md`

## 命名与定义一致性
- DDQN 定义保持不变（online argmax + target eval）。
- DQfD 定义保持不变（PER + 1-step/n-step + margin + L2）。
- strict no-fallback 保持不变（推理期无 mask/top-k/replacement/fallback/heuristic takeover）。

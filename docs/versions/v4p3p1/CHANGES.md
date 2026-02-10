# v4p3p1 代码/配置改动

## 相对 v4p3 的核心改动
- 新增配置：`configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json`
- 单变量改动（唯一行为变量）：
  - `train.forest_train_short_prob`: `0.45 -> 0.30`
- 保持不变：
  - `episodes=300`
  - `aux_admissibility_lambda=0.01`
  - strict no-fallback 推理口径

## 运行归档更新（2026-02-10）
- 新增 `self-check -> train(300 episodes) -> infer(runs=10)` 完整记录。
- 新建版本四件套：
  - `docs/versions/v4p3p1/README.md`
  - `docs/versions/v4p3p1/CHANGES.md`
  - `docs/versions/v4p3p1/RESULTS.md`
  - `docs/versions/v4p3p1/runs/README.md`

## 受影响文件清单
- `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json`
- `docs/versions/v4p3p1/README.md`
- `docs/versions/v4p3p1/CHANGES.md`
- `docs/versions/v4p3p1/RESULTS.md`
- `docs/versions/v4p3p1/runs/README.md`

## 命名与定义一致性
- DDQN 定义保持不变（online argmax + target eval）。
- DQfD 定义保持不变（PER + 1-step/n-step + margin + L2）。
- strict no-fallback 保持不变（推理期无 mask/top-k/replacement/fallback/heuristic takeover）。

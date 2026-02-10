# v4p2 版本说明

## 版本目标
- 在保持 `strict-argmax`（旧称 strict no-fallback；推理纯 `argmax(Q)`）的前提下，引入训练期可行性辅助约束（aux admissibility head）。
- 验证“训练期可行性监督”是否能降低 `argmax_inadmissible_rate` 并提升 short/long 成功率。

## 方法摘要
- 算法主干仍为 `CNN-DDQN + DQfD`（PER + 1-step/n-step TD + margin + L2）。
- 新增训练期辅助损失：`BCE(aux_logits, admissible_mask)`。
- 推理阶段不读取辅助头输出，不做 action mask/top-k/replacement/fallback。

## 关键命令
```bash
conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke
conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke
```

## 代表性运行
- iter1（aux=0.2）：
  - train: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02/train_20260210_145239`
  - infer: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730`
- iter2（aux=0.05）：
  - train: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005/train_20260210_145857`
  - infer: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter2_aux005_infer10/20260210_150353`
- iter3（aux=0.01）：
  - train: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001/train_20260210_151314`
  - infer: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter3_aux001_infer10/20260210_151812`

## 结论
- `v4p2` 已完成 `aux=0.2/0.05/0.01` 三轮单变量 smoke，CNN 在 short/long 均为 `0.0/0.0`。
- `iter3` 的 long `argmax_inadmissible_rate` 降到 `0.351`，但 short 升到 `0.560`，且成功率未提升。
- 当前仍未超过 `v4p1`（`0.1/0.0`），`v4p2` 暂不具备进入 full20 的证据。

## 下一步
- 建议将“aux-only”路线收敛为负结论，并进入 `v4p3`：仅改 `failure-aware replay`（保持 `strict-argmax` 推理口径不变）。
- 若继续在 `v4p2` 内探索，建议只做一个变量：`forest_train_dynamic_k`，并把训练拉长到 `episodes>=300` 后再评估。
- 维持 `self-check -> smoke -> full runs=20` 的时间优先流程。

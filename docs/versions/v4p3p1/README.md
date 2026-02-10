# v4p3p1 版本说明

## 版本目标
- 在 `v4p3`（300轮）基础上做单变量迭代，优先提升 long 套件能力。
- 保持 `strict-argmax`（旧称 strict no-fallback）：推理仅 `argmax(Q)`，不引入任何接管或替换。

## 方法摘要
- 继承 `v4p3`：`CNN-DDQN + DQfD + aux admissibility(λ=0.01)`。
- 本版单变量改动：`forest_train_short_prob: 0.45 -> 0.30`。
- 训练轮数保持 `episodes=300`，其余核心口径不变。

## 关键命令
```bash
conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03
conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300 --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958/models --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10
```

## 代表性运行
- train: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03/train_20260210_160958`
- infer: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044`

## 结论
- 本轮单变量下调 `short_prob` 后，最终 infer 指标退化为 `short/long=0.0/0.0`。
- 与 `v4p3` 的 `0.2/0.0` 相比，short 回落，long 仍未改善。
- 当前可判定“仅改 `forest_train_short_prob`”不是有效方向。

## 下一步
- 建议 `v4p3p2` 继续单变量：下调 `forest_train_dynamic_k`（例如 `0.2 -> 0.1`）减少课程摆动，观察 long 是否恢复。
- 若仍 `0/0`，建议转向 replay/failure-aware 机制，而非继续只改 short/long 比例。

# v4p3 版本说明

## 版本目标
- 响应“至少训练 300 轮再评估”的要求，验证 `v4p2` 路线在长训练下的真实表现。
- 保持 `strict-argmax`（旧称 strict no-fallback）：推理阶段仅 `argmax(Q)`，不做 mask/top-k/replacement/fallback。

## 方法摘要
- 继承 `v4p2`：`CNN-DDQN + DQfD`，训练期启用 admissibility aux loss。
- 本版固定 `aux_admissibility_lambda=0.01`（来自 `v4p2 iter3`），并将 `episodes=300`。
- 其余核心定义保持：DDQN Bellman、DQfD 组件、`strict-argmax` 推理口径不变。

## 关键命令
```bash
conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001
conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300 --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732/models --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10
```

## 代表性运行
- train: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001/train_20260210_152732`
- infer: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934`

## 结论
- 300 轮后，`CNN-DDQN` 在 short 有恢复（`SR=0.2`），但 long 仍为 `0.0`。
- 相比 `v4p2`（`0.0/0.0`），short 有提升；但相较 `v3p11` 的 long 能力仍明显不足。
- 当前尚不满足进入 full20 门槛评测条件（对标 Hybrid 三条件均未通过）。

## 下一步
- 建议 `v4p3p1` 做单变量：下调 short 偏置（如 `forest_train_short_prob`）或减弱动态课程 `k`，优先恢复 long SR。
- 若下一轮 smoke 出现 short/long 同升，再进入 full `runs=20` 正式门槛评估。

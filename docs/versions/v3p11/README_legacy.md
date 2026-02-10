# v3p11

## 方法（Method）
- 版本类型：`p+1`（小改动）
- 目标：在 `strict-argmax`（旧称 strict no-fallback）下，缓解 v3p10 的 long 全 timeout 崩塌，同时维持 short 不明显恶化。
- 核心改动：
  - 保留训练双套件采样与联合 checkpoint 选择；
  - 新增 short 概率线性 ramp（仅采样调度）：
    - 目标短套件概率：`forest_train_short_prob=0.35`
    - ramp：`forest_train_short_prob_ramp=0.4`
    - 即前期更偏 short（`p_short≈1.0`），在前 40% 训练进度内过渡到 0.35，后期更多 long 采样。
- 约束一致性：
  - `strict-argmax` 保持开启（推理期纯 `argmax(Q)`）；
  - 允许计算 mask 仅用于统计/诊断，但不得影响最终动作；
  - 未引入 fallback/replacement/heuristic takeover。

## 参数与命令（Params & Commands）
- 配置：`configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke.json`
- 执行命令：
```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check

conda run -n ros2py310 python train.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke
conda run -n ros2py310 python infer.py --profile repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke
```

## 结果（Smoke）
- train run:
  - `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525`
- infer run:
  - `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/infer/20260209_202232`
- KPI（`table2_kpis_mean_raw.csv`）：
  - short: CNN `成功率=0.3` vs Hybrid `0.9`
  - long: CNN `成功率=0.5` vs Hybrid `1.0`
- failure_reason（CNN）：
  - short: `timeout=6`, `reached=3`, `collision=1`
  - long: `reached=5`, `timeout=4`, `collision=1`

## 结论（Conclusion）
- v3p11 相比 v3p10：long 从 `0.0` 恢复到 `0.5`，超时崩塌 明显缓解；short 从 `0.4` 回落到 `0.3`。
- 仍未超过 Hybrid A*-MPC（short/long 均未达标）。

## 下一步（Next）
- v3p12 建议：在 v3p11 基础上只做“小步”long 强化且尽量守住 short（例如保持 ramp，微调 short_prob 到 0.4，并加轻量 long 训练样本覆盖而不改奖励）。

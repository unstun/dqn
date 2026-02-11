# v5 版本说明

- 版本类型：**Major (v+1)**
- 研究路线：`CNN-DDQN` 推理主口径：`hybrid/shielded`
- 状态：**未通过**
- 上一版本：`v4p3p1`

## 概要
- 目标：将历史表现更优的 `v3p11` checkpoint 作为主模型口径，统一在 fixed pairs20 v1 上复评测。
- 结果（最新同场对照，RL + 4 baselines）：
  - `strict-argmax`：CNN short/long SR=`0.05/0.05`
  - `hybrid/shielded`：CNN short/long SR=`0.75/0.85`
  - `Hybrid A*` short/long SR=`1.00/1.00`
  - `Hybrid A*-MPC` short/long SR=`0.95/0.90`
  - `RRT*` short/long SR=`0.95/1.00`
  - `RRT-MPC` short/long SR=`0.95/0.95`
- 结论：`hybrid/shielded` 明显优于 strict，但仍未达到最终门槛（对标 `Hybrid A*-MPC`）。

## 方法意图
- 不改算法代码，仅做“模型版本回归 + 口径标准化复评测”。
- 主结果采用 `hybrid/shielded`；同时保留 `strict-argmax` 作为学术对照。
- baseline 对照统一采用四算法同场：`Hybrid A*`、`Hybrid A*-MPC`、`RRT*`、`RRT-MPC`。

## 复现实验配置 / 命令
- 主 smoke 配置：`configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke.json`
- fixed pairs 评测配置：
  - `configs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1.json`
- 关键命令（四基线同场）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_long_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_long_pairs20_v1`

## 代表性运行
- 主模型来源（历史 checkpoint）：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- v5 strict+4baseline short：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`
- v5 strict+4baseline long：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`
- v5 hybrid+4baseline short：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`
- v5 hybrid+4baseline long：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`

## 结论
- 本版确认：`v3p11` 在 `hybrid/shielded` 下仍显著强于 strict。
- 对比 `Hybrid A*` / `Hybrid A*-MPC` / `RRT*` / `RRT-MPC` 的同场结果已补齐。
- 按最终门槛（与 `Hybrid A*-MPC` 对比）仍是 `未通过`。

## 下一步
- 在保持 `strict-argmax` 定义不变前提下，优先降低 `argmax_inadmissible_rate`，再追求 SR 提升。
- 每轮继续固定 pairs + strict/hybrid 双口径 + 四基线同场留档，避免采样漂移误导。

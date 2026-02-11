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
- 本轮补充：新增 `mid`（14–42m）训练覆盖与推理配置（先完成 pairgen + smoke 留档）。
- 结论：`hybrid/shielded` 明显优于 strict，但仍未达到最终门槛（对标 `Hybrid A*-MPC`）。

## 方法意图
- 不改算法代码，仅做“模型版本回归 + 口径标准化复评测 + mid 距离补充”。
- 主结果采用 `hybrid/shielded`；同时保留 `strict-argmax` 作为学术对照。
- baseline 对照统一采用四算法同场：`Hybrid A*`、`Hybrid A*-MPC`、`RRT*`、`RRT-MPC`。
- `mid` 采用固定 pairs20（14–42m）保证可复现、公平比较。

## 复现实验配置 / 命令
- 主 smoke 配置：
  - `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke.json`
  - `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json`
- fixed pairs 评测配置（short/long）：
  - `configs/repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1.json`
- fixed pairs 评测配置（mid）：
  - `configs/repro_20260211_forest_a_pairs_mid20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1.json`
  - `configs/repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1.json`
- 关键命令（四基线同场）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_strict_long_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --baselines hybrid_astar hybrid_astar_mpc rrt_star rrt_mpc --out repro_20260211_v5_compare4_hybrid_long_pairs20_v1`
- 诊断（仅 long，max_steps=2400≈120s，Hybrid A*-MPC 同场）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --baselines hybrid_astar_mpc --out repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1 --max-steps 2400`
- mid pairgen + smoke：
  - `conda run -n ros2py310 python infer.py --envs forest_a::mid --out repro_20260211_pairgen_forest_a_mid20 --baselines hybrid_astar_mpc --skip-rl --random-start-goal --rand-min-dist-m 14 --rand-max-dist-m 42 --rand-edge-margin-m 3 --rand-reject-unreachable --rand-reject-policy none --runs 20 --seed 125 --baseline-timeout 60 --max-steps 1200`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1 --runs 5 --out repro_20260211_v5_smoke_strict_mid_pairs5_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1 --runs 5 --out repro_20260211_v5_smoke_hybrid_mid_pairs5_v1`
- train300 后三套件 infer20（同一新模型）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_short_pairs20_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --baselines hybrid_astar_mpc --out repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --runs 20 --out repro_20260211_v5_train300_midcover_infer20_v1`
  - `conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_long_pairs20_v1 --models runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models --baselines hybrid_astar_mpc --out repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1`

## 代表性运行
- 主模型来源（历史 checkpoint）：`runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p11_smoke/train_20260209_200525/models`
- v5 strict+4baseline short：`runs/repro_20260211_v5_compare4_strict_short_pairs20_v1/20260211_094538`
- v5 strict+4baseline long：`runs/repro_20260211_v5_compare4_strict_long_pairs20_v1/20260211_094712`
- v5 hybrid+4baseline short：`runs/repro_20260211_v5_compare4_hybrid_short_pairs20_v1/20260211_095220`
- v5 hybrid+4baseline long：`runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351`
- v5 hybrid long max_steps=2400 诊断：`runs/repro_20260211_v5_compare_hybrid_long_pairs20_maxsteps2400_v1/20260211_140420`
- v5 mid pairs20 生成：`runs/repro_20260211_pairgen_forest_a_mid20/20260211_153822`
- v5 mid strict smoke：`runs/repro_20260211_v5_smoke_strict_mid_pairs5_v1/20260211_154838`
- v5 mid hybrid smoke：`runs/repro_20260211_v5_smoke_hybrid_mid_pairs5_v1/20260211_154851`
- v5 train300（midcover）：`runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840`
- v5 train300→infer20（short）：`runs/repro_20260211_v5_train300_midcover_hybrid_short_pairs20_v1/20260211_172605`
- v5 train300→infer20（mid）：`runs/repro_20260211_v5_train300_midcover_infer20_v1/20260211_164304`
- v5 train300→infer20（long）：`runs/repro_20260211_v5_train300_midcover_hybrid_long_pairs20_v1/20260211_172706`

## 结论
- 本版确认：`v3p11` 在 `hybrid/shielded` 下仍显著强于 strict。
- 对比 `Hybrid A*` / `Hybrid A*-MPC` / `RRT*` / `RRT-MPC` 的同场结果已补齐。
- mid smoke（runs=5）观察到 `strict=0/5`、`hybrid=4/5`，说明 mid 套件下仍呈现 strict/hybrid 显著分化。
- train300→infer20（runs=20）三套件结果：`CNN-DDQN SR(short/mid/long)=0.75/0.80/0.65`，对应 `Hybrid A*-MPC SR=0.95/0.95/0.90`。
- 按最终门槛（与 `Hybrid A*-MPC` 对比）仍是 `未通过`。

## 下一步
- 在保持 `strict-argmax` 定义不变前提下，优先降低 `argmax_inadmissible_rate`，再追求 SR 提升。
- 基于 `runs/repro_20260211_v5_train300_midcover_v1/train_20260211_162840/models` 做 strict/hybrid 双口径 short/mid/long 复评（避免单口径误判）。
- 若继续增大训练轮数，建议同步调整 early-stop/评测窗口，避免 300 轮配置仍在 150 轮提前停止。
- 每轮继续固定 pairs + strict/hybrid 双口径 + 四基线同场留档，避免采样漂移误导。

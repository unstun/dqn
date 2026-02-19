# v6p2p2 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v6p2`
- 本版口径：`shielded/hybrid`（推理阶段允许干预）
- 状态：**未通过最终门槛**（本轮完成 smoke 网格 + infer20 复核，尚未做 train300+full runs=20 的完整确认）

## 本版目标
- 基于 `v6p2p1`，对 `forest_reward_k_t`（每步时间惩罚系数）与 `forest_reward_k_delta`（转角变化惩罚系数）做二维网格实验。
- 在“路径更短、曲线更平滑”目标下，给出可复现的参数建议与对照数据。

## 方法概要
- 代码侧：在 `train.py` 增加 `--forest-reward-k-t`、`--forest-reward-k-delta` 参数，并传入 `AMRBicycleEnv(...)`（森林自行车环境）奖励函数。
- 实验侧：
  - smoke 网格：`k_t ∈ {0.06, 0.10, 0.14}` × `k_delta ∈ {0.8, 1.5, 2.2}`，共 9 组。
  - 快速复核：对两组候选额外执行 `infer --runs 20`。

## 关键命令
- 网格 smoke（本轮实际执行）：
  - `conda run -n ros2py310 python scripts/sweep_v6p2p2_reward_grid.py --stage smoke --k-t-values 0.06,0.1,0.14 --k-delta-values 0.8,1.5,2.2`
- 候选复核（本轮实际执行）：
  - `conda run -n ros2py310 python infer.py --profile repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke --models runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p1_kd0p8_train/train_20260219_120601/models --runs 20 --out repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20 --seed 9101`
  - `conda run -n ros2py310 python infer.py --profile repro_20260219_v6p2p2_reward_kt_kdelta_sweep_smoke --models runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_train/train_20260219_115647/models --runs 20 --out repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20 --seed 9102`

## 参数结论（本轮）
- 以“更短 + 更平滑”为主目标：**建议 `k_t=0.06`, `k_delta=1.5`**。
- 若你优先成功率（`success_rate`）而接受更长/更不平滑轨迹，可选：`k_t=0.10`, `k_delta=0.8`。

## 代表 run
- smoke 汇总：`runs/repro_20260219_v6p2p2_reward_sweep/smoke_summary_latest.csv`
- 推荐参数 smoke 代表：
  - `runs/repro_20260219_v6p2p2_reward_sweep_smoke_kt0p06_kd1p5_infer/20260219_120004/table2_kpis_mean_raw.csv`
- 20-run 复核：
  - `runs/repro_20260219_v6p2p2_reward_sweep_kt0p06_kd1p5_infer20/20260219_123827/table2_kpis_mean_raw.csv`
  - `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433/table2_kpis_mean_raw.csv`

## 下一步
1. 用 `configs/v6p2p2.json` 执行 `train300`（全量训练）+ `short/long runs=20` 完整复核。
2. 仅在 full 口径确认后，再决定是否将该参数组作为默认推荐。

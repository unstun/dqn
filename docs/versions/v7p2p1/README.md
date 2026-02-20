# v7p2p1 版本说明（失败归档）

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2`
- 本版定位：**失败归档 + 主线回退版**
- 当前主线回退目标：`v7p1`
- 状态：**已完成 smoke150（train 150 + infer runs=3）与 infer runs=20 评测，判定收益不稳定**

## 本版目标
- 按统一流程验证 `v7p2` 改动是否能稳定提升（路径更短、计算时间更短、路径更平滑）。
- 若收益不足，形成可追溯失败留档，并将主线回退到上一稳定口径（`v7p1`）。

## 关键执行记录（2026-02-20）
- 远端训练（smoke150）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2 --episodes 150 --out v7p2_es150 --device cuda --progress"`
- 远端推理（runs=20）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_es150 --runs 20 --out v7p2_es150 --progress"`
- 远端推理（runs=3，对照 smoke 口径）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2 --models v7p2_es150 --runs 3 --out v7p2_es150_eval3 --progress"`
- 对照尝试（失败）：
  - `v7p1_remote150` 在当前代码口径下执行 `runs=20` 推理时报 `obs_dim` 不兼容（`154 vs 155`），无法直接公平复跑。

## 代表 run
- 训练：`runs/v7p2_es150/train_20260220_222056`
- 推理（runs=20）：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016`
- 推理（runs=3）：`runs/v7p2_es150_eval3/20260220_223301`

## 核心结论
- `v7p2` 在本轮 `smoke150 + runs=20` 下未体现“稳定全面收益”：
  - mid 有改善，但 long 套件成功率与路径/时间仍明显劣化。
- 因此本版定义为**失败版本 `v7p2p1`**，并执行主线回退到 `v7p1`。

## 下一步（v7p2p2）
1. 在 `v7p1` 基线上做单一小改动，避免叠加变量。
2. 严格执行：`self-check -> smoke150(runs=3) -> 决策是否进入 full`。
3. 目标指标保持三条：路径更短、计算时间更短、路径更平滑。

# v6p4 版本说明

- 版本类型：**Minor（p+1）**
- 上一版本：`v6p3`
- 本版口径：`shielded/hybrid`（推理阶段允许 `top-k`（Q 值前 k 个动作重选）与 `stop override`（到达目标位置后优先停稳动作））
- 状态：**已完成配置升级与 self-check，待 smoke/full 评测**

## 本版目标
- 修复 `epsilon`（ε-greedy 探索率）衰减与训练轮数不匹配问题。
- 降低 DQfD 监督项强度，避免 `demo_lambda`（示范大间隔损失权重）过强压制 RL 自主改进。
- 提高 `obs_map_size`（下采样全局地图边长）以增强障碍分辨率。

## 方法概要
- 训练仍仅保留 `cnn-ddqn`（卷积 Double DQN），基线仍仅保留 `Hybrid A*-MPC`（混合 A* + MPC）。
- `episodes`（训练轮数）从 `300` 提升到 `3000`，并将 `eps_decay`（ε 衰减周期）改为与训练量级一致。
- `eps_decay`（ε 衰减周期）采用自适应口径：当 `eps_decay<=0` 时，按 `round(episodes * eps_decay_ratio)` 自动解析。
- `forest_demo_pretrain_steps`（示范预训练步数）与 margin 监督权重下调，减轻示范先验束缚。
- 在训练入口新增参数体检告警（仅提示不改行为），用于提前发现典型“看起来能跑、但难以收敛”的配置组合。

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --profile v6p4 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --self-check`
- smoke：
  - `conda run -n ros2py310 python train.py --profile v6p4 --episodes 300 --out v6p4_smoke300`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --models v6p4_smoke300 --runs 3 --out v6p4_smoke300`
- full：
  - `conda run -n ros2py310 python train.py --profile v6p4 --out v6p4_full3000`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --models v6p4_full3000 --runs 20 --out v6p4_full3000`

## 代表 run
- 训练：`N/A`（待运行）
- 推理：`N/A`（待运行）

## 当前结论
- 代码/配置层面已完成 `v6p4` 参数修正，尚未产出本版 KPI。
- 由于尚未执行 smoke/full，本版结论暂不涉及性能优劣，只确认“训练口径更接近标准 RL 设定”。

## 下一步
1. 先执行 `self-check -> smoke`，验证训练是否稳定推进。
2. 若 smoke 显示中期指标改善，再进入 `runs=20` 的 full 评测。
3. 将真实 `run_dir/run_json/kpi` 回填到 `RESULTS.md` 与 `runs/README.md`。

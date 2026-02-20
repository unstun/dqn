# v7p2p2 版本说明（epsilon 衰减修复尝试）

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p1`
- 本版定位：**单变量修复（ε 衰减）+ smoke 决策版**
- 当前主线：**保持 `v7p1`**
- 状态：**已完成 smoke150（train 150 + infer runs=3），未观察到稳定收益**

## 本版目标
- 修复 `epsilon`（ε-greedy 的探索率/随机动作概率）衰减与训练轮数不匹配问题。
- 在保持 `v7p1` 其余参数不变的前提下，仅调整 `eps_decay`（线性衰减轮数）验证收益。

## 关键执行记录（2026-02-20）
- 远端自检：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端训练（smoke150）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2p2 --episodes 150 --out v7p2p2_smoke150 --device cuda --progress"`
- 远端推理（runs=3）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2p2 --models v7p2p2_smoke150 --runs 3 --out v7p2p2_smoke150 --progress"`
- 对照推理（`v7p1_remote150`，runs=3）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p1 --models v7p1_remote150 --runs 3 --out v7p1_remote150_eval3 --progress"`

## 代表 run
- 训练：`runs/v7p2p2_smoke150/train_20260220_230753`
- 推理（v7p2p2，runs=3）：`runs/v7p2p2_smoke150/train_20260220_230753/infer/20260220_232053`
- 对照推理（v7p1，runs=3）：`runs/v7p1_remote150_eval3/20260220_232121`

## 核心结论
- 修复有效性（机制层面）：`eps_decay` 从 `4500 -> 200` 后，`episode=150` 时 `epsilon` 从约 `0.194` 降到约 `0.065`，衰减已明显生效。
- 收益有效性（结果层面）：本轮 smoke `runs=3` 未体现稳定正收益。
  - short 成功率从 `1.0 -> 0.667`（下降）。
  - mid 成功率持平（`0.667`），但路径长度与曲率变差。
  - long 成功率持平（`0.333`），但路径长度/时间显著变差。
- 决策：将 `v7p2p2` 记为**未通过 smoke 门**，主线继续保持 `v7p1`，准备下一轮单变量版本（`v7p2p3`）。

## 下一步
1. 基于 `v7p1` 做下一轮单变量改动，不叠加多个变量。
2. 优先改与 long 超时直接相关的量（如动作空间离散度或 progress 约束），保留 `self-check -> smoke150(runs=3) -> 决策`。
3. 若 smoke 出现清晰正向信号，再进入 full 评测（short/long 各 runs=20）。

# v7p2p3 版本说明（epsilon 衰减修复）

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p2p1`
- 本版定位：**修复 ε 衰减周期与训练轮数不匹配问题**
- 推理口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**已完成 train300（允许早停）+ infer（best）**

## 本版目标
- 修复 `epsilon`（ε-greedy 的探索率/随机动作概率）在短训练窗口内几乎不下降的问题。
- 在 `v7p1` 基线下，仅修改 `eps_decay`（线性衰减轮数）做单变量验证。

## 方法摘要
- 单一改动：`train.eps_decay: 4500 -> 260`。
- 其余训练与推理参数保持 `v7p1` 不变，避免叠加变量。

## 关键执行记录（2026-02-21）
- 远端训练（允许早停，best）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v7p2p3 --episodes 300 --out v7p2p3_train300_esbest --device cuda --progress --save-ckpt best"`
- 远端推理（best，runs=5）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile v7p2p3 --models v7p2p3_train300_esbest --out v7p2p3_train300_esbest --progress"`

## 代表 run
- 训练：`runs/v7p2p3_train300_esbest/train_20260221_003108`
- 推理：`runs/v7p2p3_train300_esbest/train_20260221_003108/infer/20260221_004529`

## 核心结论
- 机制层面：ε 衰减修复生效（`ep=260` 时 ε 从 `0.1896` 降到 `0.0200`）。
- 结果层面：本次 `runs=5` 对比 `v7p1_train300_esbest` 未体现稳定提升，尤其 short/mid 成功率下降明显。
- 决策：`v7p2p3` 归档为失败尝试，主线仍保持 `v7p1`。

## 下一步建议
1. 保留 `v7p1` 作为主线口径，避免在主线直接替换为 `v7p2p3`。
2. 下一版建议继续单变量，但优先改“探索率衰减策略”实现（如分段/自适应），避免固定线性衰减对中后期过快收缩。

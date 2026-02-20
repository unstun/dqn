# v7p2p2 - 变更

## 版本意图
- 修复 `epsilon` 衰减周期过长导致“训练全程近似高随机探索”的问题。
- 保持其他训练/推理口径不变，验证单变量收益。

## 相对 v7p2p1 的代码/配置变更
- 配置新增：
  - `configs/v7p2p2.json`
    - `train.eps_decay`: `4500 -> 200`（唯一算法相关改动）
    - 其余关键参数保持与 `v7p1` 一致。
- 复现配置新增：
  - `configs/repro_20260220_v7p2p2_eps_decay_fix.json`
    - 固化 `smoke150 + runs=3` 执行命令与关键参数。

## 文档与留档变更
- 新增版本四件套：
  - `docs/versions/v7p2p2/README.md`
  - `docs/versions/v7p2p2/CHANGES.md`
  - `docs/versions/v7p2p2/RESULTS.md`
  - `docs/versions/v7p2p2/runs/README.md`
- 版本索引同步：
  - `README.md`
  - `README.zh-CN.md`
  - `docs/versions/README.md`

## 关键参数快照
- `eps_start`: `0.2`
- `eps_final`: `0.02`
- `eps_decay`: `200`（`v7p1` 为 `4500`）
- `save_ckpt`: `best`
- 推理口径：`shielded/hybrid`（`forest_no_fallback=false`）

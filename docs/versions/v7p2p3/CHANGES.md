# v7p2p3 - 变更

## 版本意图
- 修复 `eps_decay`（线性 ε 衰减轮数）严重偏大导致训练期“长期高随机动作”的问题。

## 相对 v7p1 的具体改动
- 新增配置：`configs/v7p2p3.json`
  - `train.eps_decay`: `4500 -> 260`
  - `train.out`: `v7p2p3`
  - `infer.models/out`: `v7p2p3`
- 新增复现配置：`configs/repro_20260221_v7p2p3_eps_decay_fix.json`
  - 固化 train300（允许早停）+ infer(best) 命令。

## 变更文件
- `configs/v7p2p3.json`
- `configs/repro_20260221_v7p2p3_eps_decay_fix.json`
- `docs/versions/v7p2p3/README.md`
- `docs/versions/v7p2p3/CHANGES.md`
- `docs/versions/v7p2p3/RESULTS.md`
- `docs/versions/v7p2p3/runs/README.md`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/README.md`

## 关键参数快照
- `eps_start=0.2`
- `eps_final=0.02`
- `eps_decay=260`（旧值 `4500`）
- `save_ckpt=best`
- `episodes=300`（实际早停 `220`）

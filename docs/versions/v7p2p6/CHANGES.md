# v7p2p6 改动清单（相对 v7p2p5）

## 变更目标
- 修复并显式化三项训练基础设施：奖励缩放、梯度裁剪观测、预训练有效步数门控。

## 代码/配置改动明细

### 1) 奖励缩放/裁剪接线
- `forest_vehicle_dqn/agents.py`
  - `AgentConfig` 新增：
    - `reward_scale`
    - `reward_clip_abs`
  - `observe(...)` 入 replay 前统一应用奖励变换，覆盖在线样本与 demo 样本。

### 2) 梯度裁剪可观测化
- `forest_vehicle_dqn/agents.py`
  - 新增梯度范数计算与裁剪封装，`update()` 返回：
    - `grad_norm_pre_clip`
    - `grad_clip_hit`
  - `pretrain_on_demos()` 汇总并暴露：
    - `grad_norm_pre_clip_mean`
    - `grad_clip_hit_rate`
- `forest_vehicle_dqn/cli/train.py`
  - RL 进度日志新增：
    - `grad_norm_ema`
    - `grad_clip_hit_rate`

### 3) 预训练“有效步”口径修正
- `forest_vehicle_dqn/cli/train.py`
  - 新增参数：`--forest-demo-pretrain-min-effective-steps`
  - 预训练循环从“请求步数累加”改为“有效更新步数累加”。
  - 早停新增门控：达到 `min_effective_steps` 后才允许触发。
  - 新增“连续 0 有效更新”保护停机，避免空跑。

### 4) CLI/配置扩展
- `forest_vehicle_dqn/cli/train.py`
  - 新增参数：
    - `--reward-scale`
    - `--reward-clip-abs`
    - `--forest-demo-pretrain-min-effective-steps`
  - 参数已写入 `AgentConfig` 与 `run.json`。
- 新增配置：
  - `configs/v7p2p6.json`
  - `configs/repro_20260221_v7p2p6_foundationfix_smoke.json`

### 5) 单测新增
- `tests/test_agent_training_controls.py`
  - 覆盖奖励缩放/裁剪是否写入 replay。
  - 覆盖 `update()` 梯度观测指标返回。
  - 覆盖 `pretrain_on_demos()` 梯度统计记录。

## 受影响文件清单
- `forest_vehicle_dqn/agents.py`
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_agent_training_controls.py`
- `configs/v7p2p6.json`
- `configs/repro_20260221_v7p2p6_foundationfix_smoke.json`
- `docs/versions/v7p2p6/README.md`
- `docs/versions/v7p2p6/CHANGES.md`
- `docs/versions/v7p2p6/RESULTS.md`
- `docs/versions/v7p2p6/runs/README.md`

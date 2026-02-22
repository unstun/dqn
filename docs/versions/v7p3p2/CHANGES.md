# v7p3p2 改动清单（相对 v7p3p1）

## 变更目标
- 缓解“遇障急拐导致绕路”的策略偏差，优先减少无效大转向，再观察路径长度与时间指标。

## 代码/配置改动明细

### 1) 推理/训练替换策略（核心）
- `forest_vehicle_dqn/cli/train.py`
- `forest_vehicle_dqn/cli/infer.py`
- 新增 turn-aware 候选评分：在 `a0` 不可行时，对 admissible top-k（及 mask 候选）加入转向激进度惩罚重排。
- 新增参数：
  - `--forest-topk-turn-penalty`
- 口径：默认 `0.0` 与历史行为一致；`>0` 时更偏向平滑动作。

### 2) 单测补充
- `tests/test_v7p3p2_turn_aware_topk.py`
- 覆盖：
  - turn penalty 关闭时保持“按 Q 选 admissible”；
  - turn penalty 开启时优先平滑动作；
  - strict 模式不启用替换。

### 3) 新增配置
- `configs/v7p3p2.json`
- `configs/repro_20260222_v7p3p2_turnaware_smoke.json`
- 关键联动参数：
  - `forest_topk_turn_penalty=1.0`
  - `forest_min_progress_m=-0.01`
  - `forest_reward_k_delta=1.1`
  - `forest_train_no_progress_penalty_dist_gain=0.10`
  - `forest_train_no_progress_penalty_max=0.45`

### 4) 版本归档与索引
- 新增并回填 `docs/versions/v7p3p2/` 四件套。
- 同步更新：
  - `docs/versions/README.md`
  - `README.md`
  - `README.zh-CN.md`
  - `configs/INDEX.md`

## 受影响文件清单
- `forest_vehicle_dqn/cli/train.py`
- `forest_vehicle_dqn/cli/infer.py`
- `tests/test_v7p3p2_turn_aware_topk.py`
- `configs/v7p3p2.json`
- `configs/repro_20260222_v7p3p2_turnaware_smoke.json`
- `docs/versions/v7p3p2/README.md`
- `docs/versions/v7p3p2/CHANGES.md`
- `docs/versions/v7p3p2/RESULTS.md`
- `docs/versions/v7p3p2/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- `configs/INDEX.md`

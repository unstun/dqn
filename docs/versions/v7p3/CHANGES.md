# v7p3 改动清单（相对 v7p2p10）

## 变更目标
- 以最小改动验证“按训练套件（short/long）使用不同 no-progress 惩罚”是否能缓解 short/long 跷跷板。

## 代码/配置改动明细

### 1) 训练逻辑改动（核心）
- `forest_vehicle_dqn/cli/train.py`
- 新增参数：
  - `--forest-train-suite-no-progress-penalty`
  - `--forest-train-short-no-progress-penalty`
  - `--forest-train-long-no-progress-penalty`
- 新增逻辑：
  - 两套件训练时按 `ep_suite`（short/long）切换 `env.reward_no_progress_penalty`。
  - 无效覆写值（负数/NaN）自动回退基础惩罚。
  - `train_meta` 新增套件惩罚相关字段，便于 run 追溯。

### 2) 单测补充
- `tests/test_v7p3_adaptive_penalty.py`
- 覆盖：
  - 关闭开关时回退基础惩罚。
  - 开启开关时使用 short/long 覆写值。
  - 非法覆写值自动回退基础惩罚。

### 3) 新增配置
- `configs/v7p3.json`
- `configs/repro_20260221_v7p3_suite_penalty_smoke.json`
- 关键参数：
  - `forest_reward_no_progress_penalty=0.40`
  - `forest_train_suite_no_progress_penalty=true`
  - `forest_train_short_no_progress_penalty=0.45`
  - `forest_train_long_no_progress_penalty=0.35`

### 4) 版本归档与索引
- 新增并回填 `docs/versions/v7p3/` 四件套（已写入真实 run 路径、KPI 与失败分布）。
- 同步更新：
  - `docs/versions/README.md`
  - `README.md`
  - `README.zh-CN.md`
  - `configs/INDEX.md`

## 受影响文件清单
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_v7p3_adaptive_penalty.py`
- `configs/v7p3.json`
- `configs/repro_20260221_v7p3_suite_penalty_smoke.json`
- `docs/versions/v7p3/README.md`
- `docs/versions/v7p3/CHANGES.md`
- `docs/versions/v7p3/RESULTS.md`
- `docs/versions/v7p3/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- `configs/INDEX.md`

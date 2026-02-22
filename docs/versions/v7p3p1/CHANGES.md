# v7p3p1 改动清单（相对 v7p3）

## 变更目标
- 去除按 short/long 套件硬编码 no-progress 惩罚的做法，改为与套件标签解耦的通用自适应惩罚函数，验证泛化能力。

## 代码/配置改动明细

### 1) 训练逻辑改动（核心）
- `forest_vehicle_dqn/cli/train.py`
- 新增参数：
  - `--forest-train-adaptive-no-progress-penalty`
  - `--forest-train-no-progress-penalty-dist-gain`
  - `--forest-train-no-progress-penalty-min`
  - `--forest-train-no-progress-penalty-max`
- 新增逻辑：
  - 新增 `_compute_adaptive_no_progress_penalty(...)`，按 `dist_ratio`（起终点距离归一化）计算每回合惩罚。
  - 训练阶段在每回合 `env.reset(...)` 后动态设置 `env.reward_no_progress_penalty`。
  - `train_meta` 新增自适应惩罚统计字段（mean/min/max、dist_ratio_mean 等）用于 run 可追溯。
  - 当启用自适应惩罚时，其值优先级高于旧的 suite-specific 覆写。

### 2) 单测补充（TDD）
- `tests/test_v7p3p1_adaptive_penalty_generalized.py`
- 覆盖：
  - 关闭开关时回退基础惩罚。
  - 开启后惩罚随 `dist_ratio` 单调增大。
  - 非法输入与上下界裁剪行为。
- 回归验证：`tests/test_v7p3_adaptive_penalty.py` 继续全通过。

### 3) 新增配置
- `configs/v7p3p1.json`
- `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json`
- 关键参数：
  - `forest_reward_no_progress_penalty=0.35`
  - `forest_train_suite_no_progress_penalty=false`
  - `forest_train_adaptive_no_progress_penalty=true`
  - `forest_train_no_progress_penalty_dist_gain=0.15`
  - `forest_train_no_progress_penalty_min=0.35`
  - `forest_train_no_progress_penalty_max=0.50`

### 4) 版本归档与索引
- 新增并回填 `docs/versions/v7p3p1/` 四件套（已写入真实 run 路径、KPI 与失败分布）。
- 同步更新：
  - `docs/versions/README.md`
  - `README.md`
  - `README.zh-CN.md`
  - `configs/INDEX.md`

## 受影响文件清单
- `forest_vehicle_dqn/cli/train.py`
- `tests/test_v7p3p1_adaptive_penalty_generalized.py`
- `configs/v7p3p1.json`
- `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json`
- `docs/versions/v7p3p1/README.md`
- `docs/versions/v7p3p1/CHANGES.md`
- `docs/versions/v7p3p1/RESULTS.md`
- `docs/versions/v7p3p1/runs/README.md`
- `docs/versions/README.md`
- `README.md`
- `README.zh-CN.md`
- `configs/INDEX.md`

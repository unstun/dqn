# v8p4 变更清单（相对 v8p3）

## 1) 代码改动

### `forest_vehicle_dqn/env.py`
- `_fallback_action_short_rollout(...)`：当 `horizon_steps=h` 下找不到 `(~coll) & (min_od>=min_od_m)` 的 safe 动作时：
  - 新增 `h=1` 的 collision-first 降阶兜底：只要存在 1-step collision-free 动作，就按 `progress_dist` 最优选择（`min_od` 打破平局），避免落入“可能返回立即碰撞动作”的 clearance-only 最后兜底。

## 2) 测试改动

### `tests/test_v8p4_fallback_short_rollout_h1.py`
- 新增单测：构造“`h=30` 全撞但 `h=1` 有不撞动作”的最小 stub，确保 fallback 优先返回 1-step collision-free 动作（且不被 `min_od_m` 过严筛空）。

## 3) 配置与文档

- 新增：
  - `configs/v8p4.json`
  - `configs/repro_20260223_v8p4_fallback_h1_smoke.json`
  - `configs/repro_20260223_v8p4_smoke_failures_regression.json`
  - `configs/pairs_v8p3_smoke_failures.json`
  - `docs/versions/v8p4/`（四件套）
- 更新：
  - `configs/INDEX.md`（V8 迭代入口切换到 `v8p4`）
  - `docs/versions/README.md`、`README.md`、`README.zh-CN.md`（索引新增 `v8p4`）


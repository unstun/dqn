# v8p3 变更清单（相对 v8p2）

## 1) 代码改动

### `forest_vehicle_dqn/env.py`
- `AMRBicycleEnv.admissible_action_mask(...)`：当 `fallback_to_safe=true` 且 progress-based `out` 为空时：
  - 优先回退到 `safe=(~coll) & (min_od>=min_od_m)`；
  - 若 `safe` 为空但存在 collision-free，则回退到 `(~coll)`（放松 `min_od_m`，collision-first），避免触发可能返回碰撞动作的最后兜底。

## 2) 测试改动

### `tests/test_v8p3_collision_first_fallback.py`
- 新增单测：构造“`min_od_m` 过严导致 safe mask 为空但仍有 collision-free 动作”的场景，确保 `fallback_to_safe=true` 时 mask 非空。

## 3) 配置与文档

- 新增：
  - `configs/v8p3.json`
  - `configs/repro_20260223_v8p3_fallback_safety_smoke.json`
  - `configs/repro_20260223_v8p3_short_collision_pair_regression.json`
  - `configs/pairs_v8p2_short_collision_run2.json`
  - `docs/versions/v8p3/`（四件套）
  - `docs/plans/2026-02-23-v8p3-fallback-safety-design.md`
  - `docs/plans/2026-02-23-v8p3-fallback-safety-implementation-plan.md`
- 更新：
  - `configs/INDEX.md`（V8 迭代入口切换到 `v8p3`）
  - `docs/versions/README.md`、`README.md`、`README.zh-CN.md`（索引新增 `v8p3`）


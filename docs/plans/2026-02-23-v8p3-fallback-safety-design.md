# v8p3 设计草案：collision-first 的 fallback（避免 short 复现 collision）

日期：2026-02-23  
作者：Codex（协作）

## 1. 背景与问题复现证据

### 1.1 v8p2 现象

- `v8p2` 的 infer-only（固定 `v7p1` checkpoint）已经验证：`dijkstra8_nocorner` 能在 `SR=1.0` 前提下显著压 long 的 `avg_path_length/path_time_s`（见 `docs/versions/v8p2/RESULTS.md`）。
- 但 `v8p2` 的 train+infer smoke（`episodes=150, runs=3`）出现 **short 可复现 collision=1/3**，并在 `--seed 34` 复测下保持一致（同一对起终点）。

复现样本（来自 `table2_kpis_raw.csv`）：
- suite：`Env. (forest_a)/short`
- start/goal（cell）：`start_xy=(30,241) -> goal_xy=(119,82)`
- run：`runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027`（seed=33）

### 1.2 初步定位（高置信）

训练与推理共享同一套“admissible gating + fallback”链路：
- 优先：top-k 可行动作（`is_action_admissible(...)`）
- 其次：masked argmax（`admissible_action_mask(...)`）
- 最后：`_fallback_action_short_rollout(...)`

当前实现存在一个“碰撞兜底”的漏洞：
- `admissible_action_mask(..., fallback_to_safe=True)` 在“progress mask 为空”时，回退到 **safe mask**，但 safe mask 仍然强制 `min_od_m`（最小净空阈值）：
  - 若处于狭窄区域，可能出现 **存在 collision-free 动作，但其最小净空 < min_od_m**，导致 safe mask 为空；
- 当 mask 为空时，会调用 `_fallback_action_short_rollout(...)`，而它的最后兜底策略注释明确写了：
  - “even if it still collides”（可能选择会碰撞的动作）

因此：short 的 `collision` 高概率来源于“mask 被 `min_od_m` 筛空 → 进入最后兜底 → 允许返回碰撞动作”。

## 2. v8p3 目标（smoke 口径）

硬约束：
- 在 `success_rate≈1.0` 前提下（至少 short/mid/long smoke 全 `SR=1.0`），再谈压 `avg_path_length/path_time_s`。

次目标：
- 尽量不显著提高 `fallback_rate`（兜底触发率）与 `argmax_inadmissible_rate`（argmax 不可行动作率）。

## 3. 方案（推荐 A：工程修复，collision-first）

### 3.1 核心改动

修改 `AMRBicycleEnv.admissible_action_mask(..., fallback_to_safe=True)` 的最后回退逻辑：

- 当前（v8p2）：
  - 当 `out` 为空时：回退到 `(~coll) & (min_od >= min_od_m)`（仍强制 `min_od_m`）
- v8p3（建议）：
  1) 先回退到 `(~coll) & (min_od >= min_od_m)`（保持原语义优先）
  2) 若仍为空，但存在 `~coll`：再回退到 `(~coll)`（只保证“不碰撞”，不再强制 `min_od_m`）

直觉：当 policy 已经被筛到“无路可走”时，安全优先级应为
`no-collision > min_od_m`；否则最后兜底可能直接撞。

### 3.2 为什么不直接调参（只降 `min_od_m`）

- `min_od_m` 下调会同时改变训练/推理中的 gating 行为，属于弱可解释调参；
- 问题根因是“mask 为空时允许返回碰撞动作”，属于工程漏洞，优先修复而不是“把阈值调到让它不触发”。

## 4. 验证与回归用例

### 4.1 单元测试（必须）

构造一个最小场景：
- 存在 collision-free 动作，但全部动作 `min_od < min_od_m`
- 期待：`admissible_action_mask(..., fallback_to_safe=True)` 返回非空（至少包含 collision-free 动作）

### 4.2 行为回归（必须）

新增固定 pairs 配置，复现 v8p2 short 的 collision pair：
- 运行 `infer.py` 使用 `--rand-pairs-json` 固定起终点
- 期待：short 的 `failure_reason` 不再出现 `collision`

### 4.3 smoke（远端）

- `episodes=150, runs=3`：先跑 smoke 门（short/mid/long）
- 如 short 仍出现 `collision/timeout`：不进入 full

## 5. 非目标（本轮不做）

- 不引入新依赖。
- 不改变 reward 结构/超参（先把碰撞兜底漏洞修掉）。
- 不做大规模重构（保持最小可评审改动）。


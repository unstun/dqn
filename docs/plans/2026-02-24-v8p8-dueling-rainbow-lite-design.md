# v8p8 设计草案：Dueling + GlobalCNN-Fusion + 可行性辅助监督（目标：short/long 满足 C 门槛）

日期：2026-02-24  
目标版本：`v8p8`（patch）  
对照基线：`Hybrid A*-MPC`（baseline）  
稳定主线：`v7p1`（当前稳定主线/对照入口）

## 0) 背景与问题陈述

现状（截至 `v8p7`）：
- 已通过推理侧“接近目标速度整形”把 smoke 的 `SR` 拉回 `≈1.0`（样本内）；但 `mid/long` 的 `path_time_s` 仍落后 baseline。
- 目标从“smoke 筛查”升级到最终硬门槛 **C**：在 `short/long` 双套件、各 `runs=20` 条件下，至少同时满足：
  - `success_rate(RL) >= success_rate(Hybrid A*-MPC)`
  - `avg_path_length(RL) < avg_path_length(Hybrid A*-MPC)`
  - `path_time_s(RL) < path_time_s(Hybrid A*-MPC)`

约束（不可违反）：
- 不改 `goal_tolerance_m`（终点容差），不通过放宽“到达判定”作弊。
- 推理期策略口径必须一致：`strict-argmax` 只用于诊断；本轮结论口径默认 `shielded/hybrid`（允许 admissible gating / fallback 等），但必须显式记录。
- 必须避免 sample drift：RL 与 baseline 的比较必须在同一固定 pairs 上完成。

## 1) 本版验收口径（C 门槛）

### 1.1 固定样本（禁止漂移）

使用仓库现成固定随机起终点集合（pairs20）：
- short：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
- long：`configs/repro_20260210_forest_a_pairs_long20_v1.json`

### 1.2 最终 gate 命令（runs=20）

对 short/long 分别跑一次（同一 run 同时输出 RL 与 baseline）：

```bash
# short (runs=20, fixed pairs)
conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p8_full20_pairs_short

# long (runs=20, fixed pairs)
conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p8_full20_pairs_long
```

验收：读取两次 run 的 `table2_kpis_mean_raw.csv`，逐套件对比 `CNN-*`（或本轮选择的 DQN 变种）与 `Hybrid A*-MPC`，必须同时满足三条硬门槛。

说明：
- 不改 `goal_tolerance_m`；同时默认不改 `goal_stop_speed_m_s` / `goal_stop_delta_deg`（停止/摆正阈值），避免“更容易 reached”的隐性作弊。

## 2) 候选方案与取舍

本版采取 “DQN 变种增强 + 更强表征 + 更稳可行性” 的组合，避免引入非 DQN 家族（如 SAC/DDPG）：

### 2.1 Dueling DQN（Dueling head）

动机：
- 森林全局图像+长时域下，很多状态动作的相对优势差异不大；Dueling 的 `V(s)`（状态价值）与 `A(s,a)`（优势）分解往往更稳，减少 Q 的噪声。

定义合规：
- `Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))`（保持可辨识性）。

### 2.2 GlobalCNN-Fusion + Spatial Prior（更强全局表征）

动机：
- baseline 能较短路径+较短时间，核心在于“全局规划更直、更早避障”；RL 要达标，需要更强的全局几何建模。
- 仓库已有 `cnn_backbone=globalcnn_fusion` 与 `globalcnn_spatial_prior`（agent/goal heatmap）能力，可直接用于 DQN 家族。

### 2.3 可行性辅助监督（aux admissibility head）

动机：
- 当前大量 `argmax_inadmissible_rate/fallback_rate` 会导致“绕路/慢/不直”，使 `path_time_s` 难以下降。
- 仓库已支持 `aux_admissibility_lambda`（训练期用下一状态的 admissible action mask 做 BCE 监督），推理仍是 argmax(Q) + 既有 gating，定义上仍属于 DQN 变种增强（训练辅助任务）。

### 2.4 推理侧目标接近速度整形（v8p7）

动机：
- v8p6 诊断显示 near-goal 末段存在“必撞态”，更早减速更可能避免进入不可行动作集合为空的区域。
- 但过早刹车会拉高 `path_time_s`；本版把该模块保留但把系数调参做成可控，并通过固定 pairs 的 full20 判定是否保留。

## 3) v8p8 设计要点（实现层面约束）

### 3.1 默认行为与回滚

- 新增的网络/训练开关默认 **不影响旧版本**（只在 `v8p8` profile 打开）。
- 若 dueling/globalcnn/aux 导致训练崩溃，可回退为：
  - 保持 `globalcnn_fusion`，关闭 dueling；
  - 或保持 dueling，关闭 aux；
  - 或全部关闭回退到 `v8p7` 的推理侧策略。

### 3.2 不能作弊的检查项

每次汇报结果必须同时列出：
- `goal_tolerance_m`（必须与 baseline 一致，且不改动）
- `goal_stop_speed_m_s` / `goal_stop_delta_deg`（本轮默认固定不动；若未来要动，必须声明且视为口径变化）
- 是否启用 `forest_no_fallback`（strict vs hybrid）
- 是否启用 `forest_goal_approach_override`（v8p7 模块）
- 对比使用的 `rand_pairs_json` 路径（必须是 pairs20）

## 4) 验证路线（时间优先）

阶段 0：最小自检（只验证工程能跑）
```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

阶段 1：smoke（筛查）
- `episodes=150` 训练 + `runs=3` 推理（short/mid/long），观察 SR 与 L/T 方向。

阶段 2：full gate（C）
- 只要 smoke 显示“short/long SR 稳且 L/T 方向正确”，立刻进入 full20（pairs20），以最终门槛为唯一结论。

## 5) 风险与备选

主要风险：
- GlobalCNN-Fusion + Dueling 可能提升 SR 但导致动作更保守、`path_time_s` 反而变大；
- aux 可行性监督如果权重过大，可能让网络过度拟合 mask，牺牲路径效率；
- 推理侧 near-goal speed shaping 如果过强，可能显著拖慢 long。

备选策略（仍保持 DQN 家族）：
- 引入 `NoisyNet`（参数噪声探索）或 `distributional C51`（分布式 Q）属于 DQN 变种，但实现改动更大；优先级低于本版的 “dueling + globalcnn + aux”。


# v6p2p5 三算法差异总表（持续维护）

> 目的：明确 `CNN-DDQN`、`DDPG`、`SAC` 的实现差异，避免口径漂移。每次修改任一算法相关参数/逻辑时，同步更新本文件。

## 1. 训练范式差异

| 维度 | CNN-DDQN | DDPG | SAC |
|---|---|---|---|
| 动作空间 | 离散（`action_table` 索引） | 连续（`delta_dot, accel`） | 连续（`delta_dot, accel`） |
| 算法范式 | Q-learning（Double DQN） | Deterministic Actor-Critic | Stochastic Actor-Critic + Entropy |
| 主网络 | `Q + Q_target` | `actor + actor_target + critic + critic_target` | `actor + q1/q2 + q1_target/q2_target + alpha` |
| 探索机制 | `epsilon`-greedy | actor 输出 + 高斯噪声 | 策略分布采样 |
| demo 使用 | DQfD 损失（margin + n-step + PER） | replay demo 比例 + actor BC（`cont_bc_lambda`） | replay demo 比例 + actor BC（`cont_bc_lambda`） |

## 2. v6p2p5 隔离开关（关键）

| 开关 | CNN-DDQN | DDPG/SAC | 备注 |
|---|---|---|---|
| `learning_starts` | 使用 | 默认回落 | 全局默认仍保留 |
| `cont_learning_starts` | 不使用 | 使用 | 连续 warmup 独立 |
| `cont_demo_frac` | 不使用 | 使用 | 连续 batch demo 比例 |
| `cont_bc_lambda` | 不使用 | 使用 | 连续 actor 的 BC 正则 |
| `forest_adm_horizon/min_progress/min_od` | 使用 | 默认回落 | 全局推理阈值 |
| `cont_forest_adm_horizon/min_progress/min_od` | 不使用 | 使用 | 连续推理阈值独立 |

## 3. 推理 fallback 差异

| 维度 | CNN-DDQN | DDPG/SAC |
|---|---|---|
| fallback 入口 | top-k / admissible mask / stop override（离散） | 连续动作 admissibility 检查不通过后替换 |
| fallback 选择策略 | 离散策略链路内重选 | v6p2p5 改为“进展优先 + L2 次级” |

## 4. 冻结约束（本版承诺）

- `CNN-DDQN`：不引入连续分支参数，不接入连续 BC，不改离散网络与损失定义。
- `DDPG/SAC`：新改动仅允许落在 `cont_*` 参数或连续 rollout 逻辑。
- 任何跨分支共享参数变更（例如全局 `learning_starts`）都必须在 `CHANGES.md` 明确说明影响范围。

## 5. 更新规则（执行要求）

1. 每次改 `train.py` / `infer.py` / `continuous_agents.py` 后，检查本表是否受影响。
2. 若新增参数，必须补一行“CNN 是否生效 / 连续是否生效”。
3. 若推理策略名义变化（strict-argmax vs shielded/hybrid），必须同步更新版本 `README.md` 与 `RESULTS.md`。

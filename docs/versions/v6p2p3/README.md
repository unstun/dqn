# v6p2p3 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v6p2p2`
- 本版口径：`shielded/hybrid`（训练与推理统一）
- 对外表述（文档/论文）：`CNN-DDQN (shielded/hybrid inference)`，不宣称 `strict-argmax`
- 状态：**已运行（训练 300 轮 + short/mid/long 各 5 轮推理）**

## 本版目标
- 固定参数：`forest_reward_k_t=0.10`、`forest_reward_k_delta=0.8`。
- 统一训练/推理规则，消除 v6p2p2 中规则不一致：
  - `forest_no_fallback`（是否严格 no-fallback）统一为 `false`
  - `forest_adm_horizon`（admissibility 前看步数）统一为 `30`
  - `forest_min_progress_m`（最小前进量阈值）统一为 `0.01`
  - `forest_min_od_m`（最小净空距离）统一为 `0.02`
  - 训练侧补齐 `top-k admissible replacement`（前 k 候选可行动作替换）
  - 训练侧补齐 `stop override`（位姿到达后强制停车）
  - `no_terminate_on_stuck`（卡住是否终止）统一为 `true`

## 方法概要
- 代码侧（`train.py`）：
  - 新增参数：`--forest-topk`、`--forest-min-od-m`
  - 新增 infer 同口径动作选择 helper（含 stop override + top-k/mask 替换）
  - 将训练中多处硬编码 `min_od_m=0.0` 改为可配置 `forest_min_od_m`
  - 训练进度评估 `_eval_train_progress_suites(...)` 改为与推理一致的动作口径
- 配置侧：
  - 新增 `configs/v6p2p3.json`，统一 train/infer 关键规则参数

## 本轮执行记录（2026-02-19）
- 训练（第一次）：`conda run -n ros2py310 python train.py --profile v6p2p3`
  - 结果：触发 RL 早停，`episodes=170/300`，`stop_reason=rl_early_stop_plateau`
- 训练（第二次，满足 300 轮）：`conda run -n ros2py310 python train.py --profile v6p2p3 --rl-early-stop-patience-points 9999 --rl-early-stop-warmup-episodes 9999`
  - 结果：`episodes=300/300`，`stop_reason=completed`
- 推理：`conda run -n ros2py310 python infer.py --profile v6p2p3`
  - 结果：`short/mid/long` 三套件各 `runs=5`

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 训练/推理：
  - `conda run -n ros2py310 python train.py --profile v6p2p3`
  - `conda run -n ros2py310 python infer.py --profile v6p2p3`

## 代表 run
- 训练（早停）：`runs/v6p2p3/train_20260219_135522`
- 训练（300 轮完成）：`runs/v6p2p3/train_20260219_142104`
- 推理（基于 300 轮模型）：`runs/v6p2p3/train_20260219_142104/infer/20260219_145315`

## 本轮结果摘要（runs=5）
- short：
  - CNN-DDQN：`success_rate=0.80`，`avg_path_length=15.7615`，`path_time_s=9.3625`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=16.8724`，`path_time_s=10.0000`
- mid：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=25.3193`，`path_time_s=15.0600`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=25.1525`，`path_time_s=13.8700`
- long：
  - CNN-DDQN：`success_rate=1.00`，`avg_path_length=46.2403`，`path_time_s=28.9500`
  - Hybrid A*-MPC：`success_rate=1.00`，`avg_path_length=43.0247`，`path_time_s=22.8200`

## 下一步
1. 进入最终门槛评测：short/long 双套件各 `runs=20`。
2. 重点修复 short 成功率与 long 路径长度/时间劣化问题。
3. 按 `runs=20` 结果更新最终门槛判定。

# dqn（森林场景车辆规划 + 跟踪）

[English](README.md) | 简体中文

本仓库聚焦于森林场景的运动学车辆（Ackermann/自行车模型）环境：`forest_a`、`forest_b`、`forest_c`、`forest_d`。

默认 conda 环境：`ros2py310`。

## 快速开始（Ubuntu/bash）

下面的命令默认你在 `dqn/` 目录下运行，因此输出默认写入 `runs/`：

```bash
cd /home/sun/phdproject/dqn/dqn
```

命令有两种用法：

- 推荐（可复现/CI 友好）：保留 `conda run -n ros2py310 ...`。
- 可选（交互式终端）：先执行一次 `conda activate ros2py310`，随后直接使用 `python ...`。

自检（快速检查依赖导入/设备配置）：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

可选依赖（训练实时窗口）：

```bash
conda run -n ros2py310 python -m pip install -r requirements-optional.txt
```

## 破坏性变更（2026-02-06）

- Forest 环境内部 cost-to-go 链路已移除，改为距离语义的塑形/采样。
- Forest 训练专家支持 `hybrid_astar` 与 `astar_mpc`（`--forest-expert {auto,hybrid_astar,astar_mpc}`，其中 `auto -> hybrid_astar`）。
- Forest 自行车观测从 `11 + 2*N*N` 改为 `10 + N*N`（仅占据图通道）。
- 随机对参数由 `*-cost-m` 全部更名为 `*-dist-m`（旧参数名已删除）。
- 推理新增 short/long 比例阈值参数：
  - `--rand-short-min-dist-ratio`、`--rand-short-max-dist-ratio`
  - `--rand-long-min-dist-ratio`、`--rand-long-max-dist-ratio`

旧版观测布局/参数命名下训练出的模型与配置不再兼容。

## 训练 / 推理（推荐：配置 profile）

profile 位于 `configs/*.json`，通过 `--profile <name>` 加载：

```bash
conda run -n ros2py310 python train.py --profile forest_a_all6_300_cuda
conda run -n ros2py310 python infer.py --profile forest_a_all6_300_cuda
```

在已激活环境（`conda activate ros2py310`）下的等价命令：

```bash
python train.py --profile forest_a_all6_300_cuda
python infer.py --profile forest_a_all6_300_cuda
```

### 最新训练/推理命令（请持续维护）

最后更新：2026-02-11  
当前推荐 profile：`repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1`

```bash
conda run -n ros2py310 python train.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
```

快速 smoke（同配置，降低训练轮数用于先验筛查）：

```bash
conda run -n ros2py310 python train.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
```

固定 mid（14–42m）推理命令（strict vs hybrid，runs=20）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1
```

### 训练实时显示（pygame，仅 RL 阶段）

默认关闭；训练时显式开启：

```bash
conda run -n ros2py310 python train.py --profile repro_20260210_train_live_view_pygame_smoke --live-view --live-view-fps 0 --live-view-window-size 900 --live-view-trail-len 300 --live-view-skip-steps 1
```

- 仅在 RL 训练 `env+algo` 阶段显示；demo collect / demo pretrain 不显示。
- 手动关闭 pygame 窗口不会中断训练，会自动降级为无窗口模式。
- 若未安装 `pygame`，训练继续执行，并打印安装提示。
- 碰撞检测框默认开启（固定车体外框，按 `pose_m` 航向角旋转）；可用 `--no-live-view-collision-box` 关闭显示。

## 版本总索引（v1 → v5）

> 说明：本索引用于统一 `docs/versions/` 的重编号口径；历史目录 `v3p1`~`v3p11` 保留原记录，未纳入本轮重编号；`v4`~`v8p3` 已于 2026-02-09 清理（误混入本仓库版本链）。

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v1` | `docs/versions/v1/` | `configs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json` | `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v2` | `docs/versions/v2/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v3` | `docs/versions/v3/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | `0.5 / 0.1` | `0.9 / 1.0` | 未通过 |

### 增量版本（v3p1 → v5）

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v3p12` | `docs/versions/v3p12/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622` | `0.0 / 0.0` | `0.95 / 1.0` | 未通过 |
| `v4p1` | `docs/versions/v4p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524` | `0.1 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p2` | `docs/versions/v4p2/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3` | `docs/versions/v4p3/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934` | `0.2 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3p1` | `docs/versions/v4p3p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v5` | `docs/versions/v5/` | `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json` | `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351` | `0.75 / 0.85` | `0.95 / 0.90` | 未通过 |

- baseline-only（`--skip-rl`）输出不计入上表；请单独查看 `runs/outputs_forest_baselines/*`、`runs/repro_20260207_*` 等目录。
- 详细四件套请见 `docs/versions/README.md` 与各版本目录。

## 推荐迭代流程（时间优先）

日常 DDQN/CNN 调参建议采用两阶段流程，优先缩短反馈周期：

1. 阶段 0（自检）：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

2. 阶段 1（smoke 快速回路）：

```bash
conda run -n ros2py310 python train.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1
```

3. 阶段 2（full 门槛评测，仅在 smoke 明显改善后执行）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260211_forest_a_cnn_ddqn_v5_smoke --models runs/repro_20260211_v5_retrain_v3p11_smoke/train_20260211_080356/models --out repro_20260211_v5_full20 --runs 20
```

默认约定：若 smoke 未显示有效提升，不直接进入长时 full 全量评测。

## 最终验收门槛（short/long 双套件 + runs=20）

最终结论必须在 short/long 双套件上分别使用 `runs=20` 汇报。

使用 `table2_kpis_mean_raw.csv`，将 `CNN-DDQN` 对比 `Hybrid A*-MPC`，并同时满足：

- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

任一套件未满足任一条件，即视为未通过最终门槛。

推理口径命名说明：本仓库区分 `strict-argmax`（旧称 strict no-fallback）与 `shielded/masked/hybrid`。`strict-argmax` 指推理期纯 `argmax(Q)`（不做 masking/top-k/stop-override/replacement/启发式接管/规划器接管）；允许计算 mask 仅用于统计/诊断。若推理期启用任何干预，请按 `shielded/masked/hybrid` 命名（不得宣称 `strict-argmax`/strict no-fallback）。

### strict-argmax vs hybrid（固定 pairs 复评测模板）

为避免 random pair 漂移，建议在**固定随机样本**上同时汇报两套推理口径：

- `strict-argmax`：使用 `--forest-no-fallback`（推理纯 `argmax(Q)`）
- `hybrid/shielded`：使用 `--no-forest-no-fallback`（允许 stop-override + replacement；不启用启发式 fallback）

固定 pairs（forest_a，short/long 各 20）：

- `configs/repro_20260210_forest_a_pairs_short20_v1.json`
- `configs/repro_20260210_forest_a_pairs_long20_v1.json`

模板（复用 profile 以保证与 checkpoint 的 env/action-space 参数一致）：

```bash
PROFILE=repro_20260211_forest_a_cnn_ddqn_v5_smoke
MODELS_DIR="runs/<exp>/<train_timestamp>/models"

# strict-argmax（short）
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_strict_short_pairs20 \\
  --forest-no-fallback

# hybrid/shielded（short）
conda run -n ros2py310 python infer.py --profile "$PROFILE" --baselines \\
  --envs forest_a::short --no-rand-two-suites --random-start-goal --runs 20 \\
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \\
  --models "$MODELS_DIR" --out repro_reval_hybrid_short_pairs20 \\
  --no-forest-no-fallback
```

long 套件同理：把 `forest_a::short` 与 `pairs_short20` 改为 `forest_a::long` 与 `pairs_long20`。

## 示教数据（DQfD）

默认训练使用 `--demo-mode dqfd`（严格 DQfD 风格）：

- 优先回放（PER）+ 重要性采样（IS）权重
- 1-step TD + n-step TD + large-margin 专家损失 + L2 正则
- 不包含 behavior cloning 的 CE（以满足 DQfD 定义口径）

如需复现旧版示教稳定器行为，请使用 `--demo-mode legacy`。

如需使用 A*+MPC 专家并启用曲线优化（shortcut + 重采样 + 最小转弯半径约束 + 双圆碰撞检测），可使用
`--forest-expert astar_mpc` 配合 `--forest-astar-opt-*` 与 `--forest_mpc_*` 参数，或直接加载 profile：

```bash
conda run -n ros2py310 python train.py --profile repro_20260208_forest_a_cnn_ddqn_dqfd_astar_mpc_curveopt_300
```

论文 PDF 与 BibTeX 已归档在 `paper/dqfd_refs/`。

## 基线评估（无需 RL checkpoint）

现在 `--baselines all` 默认包含 6 个基线（固定顺序）：

1. `astar`
2. `hybrid_astar`
3. `rrt_star`
4. `astar_mpc`（`A*-MPC`）
5. `hybrid_astar_mpc`（`Hybrid A*-MPC`）
6. `rrt_mpc`（`RRT-MPC`）

CPU 运行六基线：

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_baselines --baselines all --skip-rl --runs 5 --device cpu
```

只跑 MPC 组合基线：

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_mpc_baselines --baselines astar_mpc hybrid_astar_mpc rrt_mpc --skip-rl --runs 5 --device cpu
```

旧的 `forest_baseline_mpc_*` 配置键在 infer 加载时会被忽略（已弃用）。

### 固定随机样本公平对比（规划基线）

推荐使用固定随机样本 profile，对规划基线（A* / Hybrid A* / RRT*）进行公平评测：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260206_6baselines_fair_forest_a_fixedpairs --skip-rl
```

该 profile 使用固定 start-goal 样本文件：

- `configs/repro_20260206_6baselines_fair_forest_a_pairs.json`
- （short/long 分套件固定样本）`configs/repro_20260210_forest_a_pairs_short20_v1.json`、`configs/repro_20260210_forest_a_pairs_long20_v1.json`

## 成功判定

森林自行车模型的成功条件：

- `reached_pose`：到达目标位置容差内，并且（可选）满足朝向容差
- `reached_stop`：车辆已停止且车轮几乎打直（`|v|` 与 `|delta|` 接近 0）
- `reached` / “success” == `reached_pose AND reached_stop`

实现位置：`forest_vehicle_dqn/env.py`（`AMRBicycleEnv._step_with_controls`、`_goal_pose_reached`、`_goal_stop_reached`）。

更多可运行示例与参数速查：[`runtxt.md`](runtxt.md)。

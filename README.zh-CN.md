# dqn（森林场景车辆规划 + 跟踪）

[English](README.md) | 简体中文

本仓库聚焦于森林场景的运动学车辆（Ackermann/自行车模型）环境：`forest_a`、`forest_b`、`forest_c`、`forest_d`。

默认 conda 环境：`ros2py310`。

## AI TL;DR（合同块，2026-02-21）

```text
STABLE_PROFILE=v7p1
CLAIM_REGIME=shielded/hybrid (do not label as strict-argmax)
SMOKE_GATE=train episodes=150; infer runs=3 (screening only; no final claims)
FINAL_GATE=short runs=20 + long runs=20; pass iff:
  - success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)
  - avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)
  - path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)
REPORT_ARTIFACTS:
  - table2_kpis_mean_raw.csv (mean KPIs; use for gates/tables)
  - table2_kpis_raw.csv (per-run/per-pair diagnostics; includes failure_reason)
```

## 术语（Stages）

- `self-check`：导入/设备的快速自检。
- `micro-smoke`：可选超快速回路（例如 `episodes=40`），只做 sanity；不具备可比性。
- `smoke`：标准筛查门（`episodes=150`、`runs=3`），仅用于 go/no-go。
- `full`：最终门（short/long 双套件，各 `runs=20`）。

## 仓库导航（configs + runs）

- 配置选型索引：`configs/INDEX.md`
- 配置使用说明：`configs/README.md`
- 运行产物总览：`docs/runs/README.md`
- 归档候选清单：`docs/runs/CANDIDATES_TO_ARCHIVE_20260221.md`

## 研究目标与当前状态（2026-02-24）

- 本仓库正在采用 `vibe coding` 持续迭代：小步改动、快速验证、严格回退与归档。
- 当前稳定主线版本为 `v7p1`（`configs/v7p1.json`）。
- 当前 V8 迭代候选入口为 `v8p11`（`configs/v8p11.json`）（训练优先：强化 expert+D QfD；待 train+infer smoke → 待 full gate C；仅 fixed pairs full20 通过后才允许对外宣称收益）。
- `v8p8` 为上一候选（dueling + globalcnn_fusion + 可行性辅助监督；smoke 已跑 NO-GO；full gate 尚未跑）。
- `v8p5` 为上一版（回归 PASS；infer-only：tie-break short `collision=1/3`）。
- `v8p1` 已归档为 NO-GO（navdist progress distance；smoke SR 退化）。
- `v7p2` 到 `v7p3p7` 属于非主线迭代分支的失败/探索版本，稳定结论口径仍为 `v7p1`。
- 最终目标：在公平、可复现的评测条件下，使 RL 规划器（`CNN-DDQN`）整体超过传统方法（`Hybrid A*-MPC`）。
- 核心优化方向：路径更短（`avg_path_length` 更小）、时间更短（`path_time_s` 更小）、曲线更平滑（`avg_curvature_1_m` 更小）。

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
bash scripts/self_check.sh

#（等价的显式命令）
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
conda run -n ros2py310 python game.py --self-check
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

## 版本说明（2026-02-22）

- 截至 2026-02-22，当前稳定主线仍为 `v7p1`。
- `v7p2/v7p2p1` 进行过 Markov 观测修复尝试（加入 `prev_a_n`），但收益不稳定，已归档为失败分支。
- `v7p3` 引入了两套件训练下 short/long 分离的 no-progress 惩罚；smoke 结论为 NO-GO，已按失败版本归档。
- `v7p3p1` 改为按起终点距离比例自适应 no-progress 惩罚；smoke 中 mid/long 的 SR 回升，但 path/time/smoothness 退化，已按失败版本归档。
- `v7p3p2` 引入 turn-aware top-k 替换评分以抑制遇障急拐；smoke 中 path/time 回落但 SR 明显下滑，已按失败版本归档。
- `v7p3p3` 针对 turn-aware 替换调参（`tp=0.3`, `min_prog=0.0`）；smoke 中 long SR 回升但 short=0 且出现碰撞/超时，已按失败版本归档。
- `v7p3p4` 修复 admissible gating 的 safe fallback（progress mask 为空时不再保持 inadmissible 的 `argmax(Q)`）并修复 `fallback_rate` 统计；infer-only smoke（固定 v7p3p2 checkpoint）恢复 SR 到 `0.667/0.667/1.000` 且无碰撞，但 short/mid SR 与 path/time 仍显著落后 baseline，已按失败版本归档。
- `v7p3p6` 在 `obs_map_size=128` 下加入 long 恢复向调参（`forest_topk_turn_penalty=0.3`、`forest_min_progress_m=0.0` 与 long 偏置课程）；smoke 相比 v7p3p5 将 long SR 从 `0.000` 提升到 `0.333`，但 short/long 仍未通过 baseline 门槛，已按失败版本归档。
- `v7p3p7` 在 `obs_map_size=128` 下加入 timeout-cut 调参（`forest_topk=12`、`forest_topk_turn_penalty=0.2`、`forest_min_progress_m=0.02`）；smoke 中 short/mid SR 提升到 `1.000/1.000` 且 CNN 总 timeout 从 `5` 降到 `2`，但 long 仍 `0.333` 且 `2/3 timeout`，short/long 路径与时间门槛仍未通过，已按失败版本归档。
- `v7p1` 仍作为稳定对照基线（forest bicycle 观测维度 `10 + N*N`），新模块版本在独立版本链上持续前向迭代。
- 失败留档见：`docs/versions/v7p2p1/`。

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

最后更新：2026-02-24  
当前推荐训练 profile：`v7p1`

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py --profile v7p1
```

当前 V8 smoke profiles（实验性）：

```bash
# v8p9：infer sweep smoke（fixed pairs3 子集来自 pairs20；runs=3）
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_short_smoke
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_long_smoke

# v8p8：smoke（episodes=150, runs=3）[已跑：NO-GO]
conda run -n ros2py310 python train.py --profile repro_20260224_v8p8_smoke
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p8_smoke

# v8p7：infer-only smoke（固定 v8p6 smoke checkpoint；接近目标速度整形；runs=3）
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p7_goal_approach_infer_smoke

# v8p7：train+infer smoke（episodes=150, runs=3）[待跑]
conda run -n ros2py310 python train.py --profile v8p7
conda run -n ros2py310 python infer.py --profile v8p7

# v8p5：回归（replace-ranking 消融；复现 v8p3 smoke failures；runs=2）
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_regression

# v8p5：infer-only smoke（固定 v7p1 checkpoint；replace-ranking 消融；runs=3）
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking progress_clearance_q
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p5_replace_ranking_infer_smoke --forest-replace-ranking clearance_progress_q

# v8p6：infer-only smoke（固定 v7p1 checkpoint；replace-topq；runs=3）
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 1
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p6_replace_topq_infer_smoke --forest-replace-topq 3

# v8p6：train+infer smoke（episodes=150, runs=3）[最新：NO-GO（short/mid collision=1/3）]
conda run -n ros2py310 python train.py --profile v8p6 --forest-replace-topq 3
conda run -n ros2py310 python infer.py --profile v8p6 --forest-replace-topq 3

# v8p4：回归（复现 v8p3 smoke failures：mid collision + long timeout；runs=2）
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_smoke_failures_regression

# v8p4：train+infer smoke（episodes=150, runs=3）[暂不建议：回归 FAIL]
conda run -n ros2py310 python train.py --profile repro_20260223_v8p4_fallback_h1_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p4_fallback_h1_smoke

# v8p2 参考：infer-only A/B（固定 v7p1 checkpoint）：dijkstra8_nocorner vs euclid
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke
conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_infer_smoke --forest-progress-dist-mode euclid
```

最新归档候选（用于复盘，smoke NO-GO）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke --models runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248
```

说明：训练默认会将流程日志保存到 `<run_dir>/train_flow.log`；如需关闭可加 `--no-save-train-log`。

主线对外口径（文档/论文）：`CNN-DDQN（shielded/hybrid inference）`。
- 对 `v6p2p3/v7p1` 及后续主线版本，统一按 `shielded/hybrid` 命名与汇报。
- 不将这些主线结果表述为 `strict-argmax`。

### 远端优先执行（`ubuntu-zt`）

默认策略：训练/推理（含 smoke/full）先走 `ssh ubuntu-zt`。  
只有远端不可用时，才回落本地执行。

```bash
# 1) 本地仓库 -> 远端仓库同步（本地为准）
rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  /home/sun/phdproject/dqn/dqn/ \
  ubuntu-zt:/home/sun/phdproject/dqn/dqn/

# 2) 远端执行（示例：micro-smoke 训练，episodes=40；仅 sanity）
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke --device cuda --progress"

# 3) 远端结果 -> 本地 runs/ 回传
rsync -avz \
  ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v6p2p3_smoke/ \
  /home/sun/phdproject/dqn/dqn/runs/v6p2p3_smoke/
```

micro-smoke（可选超快回路；非标准 smoke 门）：

```bash
conda run -n ros2py310 python train.py --profile v6p2p3 --episodes 40 --out v6p2p3_smoke
conda run -n ros2py310 python infer.py --profile v6p2p3 --models v6p2p3_smoke --runs 3 --out v6p2p3_smoke
```

固定 mid（14–42m）推理命令（strict vs hybrid，runs=20，诊断消融用途）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_strict_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v5_reval_v3p11_hybrid_mid_pairs20_v1
```

v6p1 timeout-tune 固定 pairs 推理命令（hybrid/shielded，runs=20；profile 内已固定 checkpoint）：

```bash
# NOTE: v6p1 的 long/mid gating 会让 short 回退；short 仍沿用 v6。
conda run -n ros2py310 python infer.py --profile repro_20260211_v6_timeout_tune_hybrid_short_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_mid_pairs20_v1
conda run -n ros2py310 python infer.py --profile repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1
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

### 交互式点目标 game（pygame）

在地图上鼠标左键点 goal（目标点），选择规划器，然后用 `mpc` 跟踪规划路径。

```bash
conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1
```

规划器快捷键：`1`=hybrid A*，`2`=RRT*，`3`=grid A*，`4`=cnn-ddqn（需要 `--rl-checkpoint <path>`）。  
其他：`R` 重置，`SPACE` 暂停，`P` 重新规划。

## `runs/` 目录结构与工件定位

输出路径规则：
- 若 `--out <name>` 为纯名称，则输出写入 `runs/<name>/`。
- 若 `--out <path>` 为路径，则输出直接写入 `<path>/`。

典型结构（train + 嵌套 infer）：

```text
runs/<out>/
  latest.txt
  train_YYYYMMDD_HHMMSS/
    configs/run.json
    train_flow.log
    models/
    infer/
      latest.txt
      YYYYMMDD_HHMMSS/
        configs/run.json
        table2_kpis_mean_raw.csv
        table2_kpis_raw.csv
```

说明：
- 机器/脚本解析优先使用 `*_raw.csv`。非 raw 的 `table2_kpis_mean.csv` 使用更“可读”的列名（不利于稳定解析）。
- 最终门槛与对比表读取 `table2_kpis_mean_raw.csv`；失败分布与诊断读取 `table2_kpis_raw.csv`。

### KPI 字段字典（`table2_kpis_*` 列，最小必需）

- `success_rate`：成功率，范围 `[0,1]`（越大越好）。
- `avg_path_length`：平均路径长度（米，越小越好）。
- `path_time_s`：轨迹执行时间（秒，越小越好）。
- `avg_curvature_1_m`：平均曲率（`1/m`，越小越平滑）。
- `planning_time_s`：规划时间（秒，越小越好）。
- `tracking_time_s`：跟踪/控制时间（秒，越小越好）。
- `inference_time_s`：策略推理时间（秒，仅 RL；越小越好）。
- `argmax_inadmissible_rate`：`argmax(Q)` 不可行的比例（诊断指标）。
- `fallback_rate`：推理期 fallback/override 触发比例（诊断指标；在 `strict-argmax` 口径下理论应为 `0`）。
- `failure_reason`：失败原因标签（仅在 `table2_kpis_raw.csv` 中提供）。

## 版本总索引（v1 → v8p11）

> 说明：本索引用于统一 `docs/versions/` 的重编号口径；历史目录 `v3p1`~`v3p11` 保留原记录，未纳入本轮重编号；早期误混入版本链已于 2026-02-09 清理。当前稳定主线为 `v7p1`，`v7p2/v7p2p1/v7p2p2/v7p2p3/v7p2p4/v7p2p5/v7p2p6/v7p2p7/v7p2p8/v7p2p9/v7p2p10/v7p3/v7p3p1/v7p3p2/v7p3p3/v7p3p4/v7p3p6/v7p3p7` 为已归档迭代分支；`v8p11` 为当前 V8 迭代候选入口（训练侧强化：待 train+infer smoke → 待 full gate C），`v8p10` 为上一候选（infer-only sweep smoke 已跑；baseline gap；full 暂不建议）。

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v1` | `docs/versions/v1/` | `configs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke.json` | `runs/repro_20260208_forest_a_cnn_ddqn_strict_no_fallback_v1_smoke/train_20260209_002017` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v2` | `docs/versions/v2/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v2_smoke/train_20260209_083246` | `0.0 / 0.0` | `1.0 / 1.0` | 未通过 |
| `v3` | `docs/versions/v3/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3_smoke_fast4pre_h20mp0_ms1200/20260209_123403` | `0.5 / 0.1` | `0.9 / 1.0` | 未通过 |

### 增量版本（v3p1 → v8p11）

| 版本 | 目录 | 主 config | 关键 run | 最佳 SR（CNN short/long） | 基线 SR（Hybrid short/long） | 状态 |
|---|---|---|---|---|---|---|
| `v3p12` | `docs/versions/v3p12/` | `configs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_smoke_c_fast.json` | `runs/repro_20260209_forest_a_cnn_ddqn_strict_no_fallback_v3p12_full20_cfast/20260210_024622` | `0.0 / 0.0` | `0.95 / 1.0` | 未通过 |
| `v4p1` | `docs/versions/v4p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524` | `0.1 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p2` | `docs/versions/v4p2/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p2_smoke_iter1_aux02_infer10/20260210_145730` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3` | `docs/versions/v4p3/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3_smoke300_iter1_ep300_aux001_infer10/20260210_155934` | `0.2 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v4p3p1` | `docs/versions/v4p3p1/` | `configs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300.json` | `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p3p1_smoke300_iter1_sp03_infer10/20260210_164044` | `0.0 / 0.0` | `0.9 / 1.0` | 未通过 |
| `v5` | `docs/versions/v5/` | `configs/repro_20260211_forest_a_cnn_ddqn_v5_smoke_midcover_v1.json` | `runs/repro_20260211_v5_compare4_hybrid_long_pairs20_v1/20260211_095351` | `0.75 / 0.85` | `0.95 / 0.90` | 未通过 |
| `v6` | `docs/versions/v6/` | `configs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6_timeout_tune_hybrid_long_pairs20_v1/20260211_214602` | `0.90 / 0.70` | `0.95 / 0.90` | 未通过 |
| `v6p1` | `docs/versions/v6p1/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2` | `docs/versions/v6p2/` | `configs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1.json` | `runs/repro_20260211_v6p1_timeout_tune_hybrid_long_pairs20_v1/20260212_003414` | `0.70 / 0.95` | `0.95 / 0.90` | 未通过 |
| `v6p2p2` | `docs/versions/v6p2p2/` | `configs/v6p2p2.json` | `runs/repro_20260219_v6p2p2_reward_sweep_kt0p1_kd0p8_infer20/20260219_123433` | `0.75 / 0.55` | `0.95 / 1.00` | 未通过（待 full） |
| `v6p2p3` | `docs/versions/v6p2p3/` | `configs/v6p2p3.json` | `runs/v6p2p3/train_20260219_142104/infer/20260219_145315` | `0.80 / 1.00` | `1.00 / 1.00` | 已运行（runs=5，待 full20） |
| `v7p1` | `docs/versions/v7p1/` | `configs/v7p1.json` | `runs/v7p1_train300_esbest/train_20260221_010743/infer/20260221_011927` | `1.00 / 1.00` | `1.00 / 1.00` | 稳定主线（runs=5，待 full20） |
| `v7p2` | `docs/versions/v7p2/` | `configs/v7p2.json` | `runs/v7p2_smoke/train_20260220_211732/infer/20260220_212137` | `1.00 / 1.00` | `1.00 / 1.00` | 已运行（micro-smoke：episodes=40, runs=3） |
| `v7p2p1` | `docs/versions/v7p2p1/` | `configs/repro_20260220_v7p2p1_rollback_v7p1.json` | `runs/v7p2_es150/train_20260220_222056/infer/20260220_223016` | `0.85 / 0.65` | `0.95 / 1.00` | 失败归档，已回退到 `v7p1` |
| `v7p2p2` | `docs/versions/v7p2p2/` | `configs/repro_20260221_v7p2p2_globalcnn_smoke.json` | `runs/v7p2p2_globalcnn_smoke/train_20260221_171611/infer/20260221_172943` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p3` | `docs/versions/v7p2p3/` | `configs/repro_20260221_v7p2p3_globalcnn_fusion_smoke.json` | `runs/v7p2p3_globalcnn_fusion_smoke/train_20260221_174334/infer/20260221_180256` | `0.333 / 0.667` | `1.00 / 1.00` | 失败归档（smoke 不达门，主线保持 `v7p1`） |
| `v7p2p4` | `docs/versions/v7p2p4/` | `configs/repro_20260221_v7p2p4_globalcnn_spatialprior_smoke.json` | `runs/v7p2p4_globalcnn_spatialprior_smoke/train_20260221_182908/infer/20260221_183926` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（smoke 不达门，保持当前代码并继续前向迭代） |
| `v7p2p5` | `docs/versions/v7p2p5/` | `configs/repro_20260221_v7p2p5_globalcnn_fusionnorm_smoke.json` | `runs/v7p2p5_globalcnn_fusionnorm_smoke/train_20260221_202023/infer/20260221_203626` | `0.333 / 0.667` | `1.00 / 1.00` | 失败归档（smoke 退化，不回退代码并继续前向迭代） |
| `v7p2p6` | `docs/versions/v7p2p6/` | `configs/repro_20260221_v7p2p6_foundationfix_smoke.json` | `runs/v7p2p6_foundationfix_smoke/train_20260221_211603/infer/20260221_213248` | `1.000 / 0.000` | `1.00 / 1.00` | 失败归档（short 改善但 long 崩塌，继续前向迭代） |
| `v7p2p7` | `docs/versions/v7p2p7/` | `configs/repro_20260221_v7p2p7_gradclip_recover_smoke.json` | `runs/v7p2p7_gradclip_recover_smoke/train_20260221_215452/infer/20260221_221008` | `0.333 / 0.333` | `1.00 / 1.00` | 失败归档（long 有恢复但 short 退化，继续前向迭代） |
| `v7p2p8` | `docs/versions/v7p2p8/` | `configs/repro_20260221_v7p2p8_bold_dynamic_expert_smoke.json` | `runs/v7p2p8_bold_dynamic_expert_smoke/train_20260221_225358/infer/20260221_230426` | `0.000 / 1.000` | `1.00 / 1.00` | 失败归档（long 恢复到 1.0，但 short 崩塌到 0.0，继续前向迭代） |
| `v7p2p9` | `docs/versions/v7p2p9/` | `configs/repro_20260221_v7p2p9_ablate_expert_smoke.json` | `runs/v7p2p9_ablate_expert_smoke/train_20260221_231402/infer/20260221_232825` | `0.667 / 0.000` | `1.00 / 1.00` | 失败归档（short 回升但 long 崩塌，继续前向迭代） |
| `v7p2p10` | `docs/versions/v7p2p10/` | `configs/repro_20260221_v7p2p10_penalty035_smoke.json` | `runs/v7p2p10_penalty035_smoke/train_20260221_234022/infer/20260221_235340` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（long 回升但 short 路径与平滑性退化，继续前向迭代） |
| `v7p3` | `docs/versions/v7p3/` | `configs/repro_20260221_v7p3_suite_penalty_smoke.json` | `runs/v7p3_suite_penalty_smoke/train_20260222_012415/infer/20260222_014023` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（short/mid 局部改善但 long path/time 退化，未过 smoke 门） |
| `v7p3p1` | `docs/versions/v7p3p1/` | `configs/repro_20260222_v7p3p1_adaptive_penalty_smoke.json` | `runs/v7p3p1_adaptive_penalty_smoke/train_20260222_091303/infer/20260222_093552` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（mid/long SR 提升至 1.0，但 path/time/smoothness 全面退化） |
| `v7p3p2` | `docs/versions/v7p3p2/` | `configs/repro_20260222_v7p3p2_turnaware_smoke.json` | `runs/v7p3p2_turnaware_smoke/train_20260222_101744/infer/20260222_103842` | `0.333 / 0.333` | `1.00 / 1.00` | 失败归档（路径/时间有所回落，但三套件 SR 显著下降，未过 smoke 门） |
| `v7p3p3` | `docs/versions/v7p3p3/` | `configs/repro_20260222_v7p3p3_infergate_smoke.json` | `runs/v7p3p3_infergate_smoke/train_20260222_112955/infer/20260222_114657` | `0.000 / 0.667` | `1.00 / 1.00` | 失败归档（long SR 回升，但 short=0 且出现碰撞/超时，未过 smoke 门） |
| `v7p3p4` | `docs/versions/v7p3p4/` | `configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json` | `runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513` | `0.667 / 1.000` | `1.00 / 1.00` | 失败归档（safe fallback 补丁修复碰撞回潮；但 short/mid SR 仍落后 baseline，且 path/time 更差；infer-only smoke） |
| `v7p3p6` | `docs/versions/v7p3p6/` | `configs/repro_20260222_v7p3p6_obsmap128_tune_smoke.json` | `runs/v7p3p6_obsmap128_tune_smoke/train_20260222_215007/infer/20260222_223831` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（long 从 0.000 回升到 0.333，但 short/long 仍未过门） |
| `v7p3p7` | `docs/versions/v7p3p7/` | `configs/repro_20260222_v7p3p7_obsmap128_timeoutcut_smoke.json` | `runs/v7p3p7_obsmap128_timeoutcut_smoke/train_20260222_230248/infer/20260222_235329` | `1.000 / 0.333` | `1.00 / 1.00` | 失败归档（short/mid SR 升至 1.0 且 CNN 总 timeout 从 5 降到 2；但 long 仍 2/3 timeout，short/long path-time 仍落后 baseline） |
| `v8p1` | `docs/versions/v8p1/` | `configs/v8p1.json` | `runs/v8p1_navdist_smoke/train_20260223_021339/infer/20260223_023932` | `0.667 / 0.333` | `1.00 / 1.00` | 失败归档（navdist progress distance；smoke SR 退化） |
| `v8p2` | `docs/versions/v8p2/` | `configs/v8p2.json` | `runs/v8p2_costmap_smoke/train_20260223_104408/infer/20260223_110027` | `0.667 / 1.000` | `1.00 / 1.00` | smoke 已跑（mid/long=1.0；short=2/3 collision；暂不 full） |
| `v8p3` | `docs/versions/v8p3/` | `configs/v8p3.json` | `runs/v8p3_fallback_safety_smoke/train_20260223_125609/infer/20260223_131153` | `1.000 / 0.667` | `1.00 / 1.00` | 失败归档（smoke：mid collision=1/3；long timeout=1/3） |
| `v8p4` | `docs/versions/v8p4/` | `configs/v8p4.json` | `runs/v8p4_smoke_failures_regression/20260223_142739` | `N/A / N/A` | `N/A / N/A` | 失败归档（回归 FAIL：collision+timeout；暂不 smoke） |
| `v8p5` | `docs/versions/v8p5/` | `configs/v8p5.json` | `runs/v8p5_replace_ranking_infer_smoke/20260223_172217` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：`q` PASS；tie-break short `collision=1/3`（NO-GO）；train+infer smoke 未跑 |
| `v8p6` | `docs/versions/v8p6/` | `configs/v8p6.json` | `runs/v8p6_replace_topq_infer_smoke/20260223_185628` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：topq=1/2/3 均 PASS（推荐 topq=3）；train+infer smoke NO-GO（short/mid collision=1/3） |
| `v8p7` | `docs/versions/v8p7/` | `configs/v8p7.json` | `runs/v8p7_goal_approach_infer_smoke/20260223_230524` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only：接近目标速度整形 PASS（SR=1.0）；train+infer smoke 待跑 |
| `v8p8` | `docs/versions/v8p8/` | `configs/v8p8.json` | `runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556` | `0.667 / 1.000` | `1.00 / 1.00` | smoke 已跑（NO-GO；short SR 低于 baseline；mid/long path/time 劣于 baseline） |
| `v8p9` | `docs/versions/v8p9/` | `configs/v8p9.json` | `runs/v8p9_infer_sweep_short_pairs3_smoke/20260224_114743` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only sweep smoke 已跑（pairs3：SR=1.0 可达；但 path/time 仍落后 baseline；full 暂不建议） |
| `v8p10` | `docs/versions/v8p10/` | `configs/v8p10.json` | `runs/v8p10_sweep_long_w2p0/20260224_135035` | `1.000 / 1.000` | `1.00 / 1.00` | infer-only sweep smoke 已跑（pairs3：SR=1.0 可达；w_clearance sweep 有回落但仍落后 baseline；full 暂不建议） |
| `v8p11` | `docs/versions/v8p11/` | `configs/v8p11.json` | `N/A` | `N/A / N/A` | `N/A / N/A` | 进行中（训练侧强化：待 train+infer smoke → 待 full gate C） |

- baseline-only（`--skip-rl`）输出不计入上表；请单独查看 `runs/outputs_forest_baselines/*`、`runs/repro_20260207_*` 等目录。
- 详细四件套请见 `docs/versions/README.md` 与各版本目录。

## 严谨性与反作弊规则（强制）

- RL 与基线对比必须使用同一环境、同一套件、同一组固定起终点样本（禁止样本漂移）。
- 宣称收益时必须保持评测预算与口径一致；最终结论必须使用 short/long 双套件且各 `runs=20`。
- smoke 结果（`episodes=150`、`runs=3`）仅用于筛查，不得作为最终结论。
- 推理策略命名必须与实现一致（`strict-argmax` vs `shielded/masked/hybrid`），禁止隐藏干预后仍宣称 strict。
- 禁止挑结果：失败版本/失败运行必须归档；结果缺失必须写 `N/A` 并说明原因。
- 任何结论必须可追溯到工件：命令行、`run_dir`、`run.json`、`table2_kpis_mean_raw.csv`。

## 推荐迭代流程（版本优先）

每次做新版本默认采用以下流程：

1. 版本前快照（强制）：
- 先确保工作区 clean：`git status`
- 改动前先快照并推送：
```bash
git add -A
git commit -m "snapshot: pre-<version>"
git push origin <branch>
```

2. 仅做一个小版本改动（单一目的）。

3. 固定 smoke 门：
```bash
conda run -n ros2py310 python train.py --profile <candidate> --episodes 150 --out <version>_smoke150 --device cuda
conda run -n ros2py310 python infer.py --profile <candidate> --models <version>_smoke150 --runs 3 --out <version>_smoke150
```

4. Go/No-Go 规则：
- 如果 smoke 没有明显收益，不进入更长 full 评测。
- 保持当前最新代码（不回退），并继续做单一目的的前向迭代。

5. 立即归档：
- 在 `docs/versions/<version>/` 写四件套，记录命令、run 路径、KPI、失败原因。
- 下一轮按 `<version+1>` 继续（例如 `v7p3p5`）。

## 最终验收门槛（short/long 双套件 + runs=20）

最终结论必须在 short/long 双套件上分别使用 `runs=20` 汇报。

使用 `table2_kpis_mean_raw.csv`，将 `CNN-DDQN` 对比 `Hybrid A*-MPC`，并同时满足：

- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

同时必须在同一结果表中报告平滑性指标 `avg_curvature_1_m`（越小越平滑）。该指标是版本筛选与优化方向中的强制汇报项。

任一套件未满足任一条件，即视为未通过最终门槛。

推理口径命名说明：本仓库区分 `strict-argmax`（旧称 strict no-fallback）与 `shielded/masked/hybrid`。`strict-argmax` 指推理期纯 `argmax(Q)`（不做 masking/top-k/stop-override/replacement/启发式接管/规划器接管）；允许计算 mask 仅用于统计/诊断。若推理期启用任何干预，请按 `shielded/masked/hybrid` 命名（不得宣称 `strict-argmax`/strict no-fallback）。当前主线对外口径为 `shielded/hybrid`。

### strict-argmax vs hybrid（固定 pairs 复评测模板，仅诊断用途）

为避免 random pair 漂移，建议在**固定随机样本**上同时汇报两套推理口径。本节用于消融/诊断，不作为当前主线版本的主声明模板。

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

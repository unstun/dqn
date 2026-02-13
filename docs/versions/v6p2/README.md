# v6p2 版本说明

- 版本类型：**Patch（p+1）**
- 研究路线：`CNN-DDQN` 推理主口径：`hybrid/shielded`
- 状态：**未通过（结果继承 v6p1；本版无新实验）**
- 上一版本：`v6p1`

## 本版目标（仓库规范化）
- 提供“一键自检”与静态一致性检查入口，降低 README/profile 漂移导致的不可复现问题。
- 增补交互式 `game.py`（pygame 目标点点击 demo），便于快速演示规划+跟踪链路。

## 方法概要
- 新增 `scripts/repo_doctor.py`：静态检查 docs/config 关键一致性（不跑训练/推理）。
- 新增 `scripts/self_check.sh`：串联 `repo_doctor + train/infer/game --self-check` 的一键自检入口。
- 新增可复现记录：`configs/repro_20260213_repo_normalize_selfcheck_v1.json`（仅记录命令与检查项；不是 profile）。

## 自检命令（推荐）
- 一键自检：
  - `bash scripts/self_check.sh`
- 仅静态检查（严格模式）：
  - `conda run -n ros2py310 python scripts/repo_doctor.py --strict`
- 交互式 demo（需要可选依赖 `requirements-optional.txt` 安装 pygame）：
  - `conda run -n ros2py310 python game.py --profile repro_20260212_interactive_game_forest_a_v1`

## 结果说明
- 本版未新增训练/推理实验；short/long KPI 与门槛检查**沿用 v6p1**，详见：
  - `docs/versions/v6p2/RESULTS.md`
  - `docs/versions/v6p2/runs/README.md`


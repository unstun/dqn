# v6p2 - 变更

## 版本意图
- 在不改 RL/规划算法口径的前提下，补齐“可审查、可自检、可复现”的仓库工程化入口，降低日常迭代的审查成本。

## 相对 v6p1 的具体变更
- 新增交互式点目标 demo：
  - `game.py`（命令行入口）
  - `forest_vehicle_dqn/cli/game.py`（pygame 交互实现；支持 `--self-check`）
  - `configs/repro_20260212_interactive_game_forest_a_v1.json`
  - `configs/repro_20260212_interactive_game_ui_cn_v1.json`
- 新增仓库自检/静态检查入口：
  - `scripts/self_check.sh`（一键自检）
  - `scripts/repo_doctor.py`（静态一致性检查）
  - `configs/repro_20260213_repo_normalize_selfcheck_v1.json`（自检可复现记录；非 profile）
- 文档同步：
  - `README.md`、`README.zh-CN.md` 增补 `scripts/self_check.sh` 与 `game.py --self-check`。
  - `configs/README.md` 增补 `game.py` profile（读取 `game` 段）的说明与示例。

## 变更文件清单
- `game.py`
- `forest_vehicle_dqn/cli/game.py`
- `scripts/self_check.sh`
- `scripts/repo_doctor.py`
- `configs/repro_20260212_interactive_game_forest_a_v1.json`
- `configs/repro_20260212_interactive_game_ui_cn_v1.json`
- `configs/repro_20260213_repo_normalize_selfcheck_v1.json`
- `configs/README.md`
- `README.md`
- `README.zh-CN.md`
- `docs/versions/v6p2/README.md`
- `docs/versions/v6p2/CHANGES.md`
- `docs/versions/v6p2/RESULTS.md`
- `docs/versions/v6p2/runs/README.md`
- `docs/versions/README.md`


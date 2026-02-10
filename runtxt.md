# Run examples + flag reference（可运行示例 + 参数速查）

This file collects runnable examples and a lightweight flag reference for `train.py` and `infer.py`.
本文件收集可直接运行的示例命令，并提供 `train.py` / `infer.py` 的常用参数速查。

## Full flag list（完整参数列表）

```bash
conda run -n ros2py310 python train.py --help
conda run -n ros2py310 python infer.py --help
```

## Examples（示例）

### Self-check（自检）

Fast sanity check of imports/device.
快速检查依赖导入/设备配置。

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### Train / infer with a profile（使用 profile 训练/推理）

Profiles live under `configs/*.json` and are loaded via `--profile <name>`.
profile 位于 `configs/*.json`，通过 `--profile <name>` 加载。

```bash
conda run -n ros2py310 python train.py --profile forest_a_all6_300_cuda
conda run -n ros2py310 python infer.py --profile forest_a_all6_300_cuda
```

### Baseline-only eval（仅基线评估）

Planner-only baselines (A* + Hybrid A* + RRT*) on CPU.
纯规划基线（A* + Hybrid A* + RRT*），CPU 运行。

```bash
conda run -n ros2py310 python infer.py --envs forest_a --out outputs_forest_baselines --baselines all --skip-rl --runs 5 --device cpu
```

### Fixed-pair fair profile（固定随机样本公平对比）

Run planner baselines on frozen random start-goal pairs.
在固定随机样本上运行规划基线。

```bash
conda run -n ros2py310 python infer.py --profile repro_20260206_6baselines_fair_forest_a_fixedpairs --skip-rl
```

Equivalent explicit flags (key parts):
等价关键参数（简写）：

```bash
conda run -n ros2py310 python infer.py \
  --envs forest_a --skip-rl --baselines all \
  --random-start-goal --runs 30 \
  --rand-pairs-json configs/repro_20260206_6baselines_fair_forest_a_pairs.json
```

## Selected flags（常用参数）

### Forest breaking migration（Forest 破坏性迁移）

- `--forest-rand-min-cost-m` / `--forest-rand-max-cost-m` -> `--forest-rand-min-dist-m` / `--forest-rand-max-dist-m`
- `--rand-min-cost-m` / `--rand-max-cost-m` -> `--rand-min-dist-m` / `--rand-max-dist-m`
- `--rand-short-min-cost-m` / `--rand-short-max-cost-m` -> `--rand-short-min-dist-m` / `--rand-short-max-dist-m`
- `--rand-long-min-cost-m` / `--rand-long-max-cost-m` -> `--rand-long-min-dist-m` / `--rand-long-max-dist-m`
- Forest expert now: `--forest-expert {auto,hybrid_astar}` (`auto` maps to `hybrid_astar`).
- Infer suite ratio flags:
  - `--rand-short-min-dist-ratio`, `--rand-short-max-dist-ratio`
  - `--rand-long-min-dist-ratio`, `--rand-long-max-dist-ratio`

### Shared（通用）

- `--config <path>`: JSON config file (CLI flags override). / JSON 配置文件（命令行参数会覆盖配置）。
- `--profile <name>`: Profile under `configs/` (e.g. `forest_a_all6_300_cuda` -> `configs/forest_a_all6_300_cuda.json`). / `configs/` 下的 profile。
- `--envs forest_a forest_b ...`: Evaluate/train a subset of envs. / 指定训练/评估的环境子集。
- `--out <name|path>`: Output experiment name/dir (bare names go under `--runs-root`). / 输出实验名/目录。
- `--runs-root <path>`: Root dir for bare experiment names (default: `runs/`). / 纯名称实验的根目录（默认：`runs/`）。
- `--timestamp-runs` / `--no-timestamp-runs`: Toggle timestamped subdirs (default: enabled). / 是否使用时间戳子目录（默认开启）。

### Train-only（仅训练）

- `--rl-algos <...>`: Which RL algorithms to train (supports `all`). / 选择训练的 RL 算法（支持 `all`）。
- `--episodes N`: Number of episodes. / 训练回合数。
- `--max-steps N`: Max steps per episode. / 单回合最大步数。
- `--device {auto,cpu,cuda}` + `--cuda-device K`: Torch device selection. / Torch 设备选择。
- `--self-check`: Print runtime info and exit. / 打印运行环境信息后退出。

### Infer-only（仅推理/评估）

- `--models <name|path>`: Where to load models from (experiment/run/models dir). / 模型来源（实验目录/运行目录/models 目录）。
- `--runs N`: Averaging runs for stochastic methods. / 对随机方法做 N 次平均。
- `--baselines hybrid_astar rrt_star` (or `all`): Enable planner baselines. / 启用规划基线。
- `--skip-rl`: Skip RL agents (baseline-only eval). / 跳过 RL（只跑基线）。
- `--baseline-timeout S`: Planner timeout in seconds. / 规划器超时（秒）。
- `--hybrid-max-nodes N`: Hybrid A* node budget. / Hybrid A* 节点预算。
- `--rrt-max-iter N`: RRT* iteration budget. / RRT* 迭代预算。
- `--rand-pairs-json <path>`: Load fixed random start-goal pairs for fair comparison. / 载入固定随机起终点样本用于公平对比。

### forest_a planner baselines, 20 runs, grouped figures（forest_a 规划基线 20 次 + 分组出图）

Run planner baseline curves on `forest_a` only, with random start-goal screening by Hybrid A*, and export grouped figures with 4 runs per image.
仅在 `forest_a` 上运行规划基线曲线，起终点由 Hybrid A* 做可达性筛选，并按每张图 4 个 run 分组导出。

```bash
conda run -n ros2py310 python infer.py --profile repro_20260207_forest_a_6baselines_20runs_group4
```

Expected grouped outputs (under the run output dir):
预期分组图输出（位于该次 run 输出目录）：

- `fig12_paths_forest_a_runs_00_03.png`
- `fig12_paths_forest_a_runs_04_07.png`
- `fig12_paths_forest_a_runs_08_11.png`
- `fig12_paths_forest_a_runs_12_15.png`
- `fig12_paths_forest_a_runs_16_19.png`
- `fig13_controls_forest_a_runs_00_03.png`
- `fig13_controls_forest_a_runs_04_07.png`
- `fig13_controls_forest_a_runs_08_11.png`
- `fig13_controls_forest_a_runs_12_15.png`
- `fig13_controls_forest_a_runs_16_19.png`

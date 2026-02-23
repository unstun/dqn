# V8p2 实施计划：Costmap Dijkstra progress distance（dijkstra8_nocorner）

日期：2026-02-23  
作者：Codex（协作）

## 0. 前置检查（强制）

1) 本地 GitHub 快照：`git status` clean，且 `git push` 成功（作为可回滚点）。  
2) 远端运行前同步：`rsync` 本地覆盖远端（不包含 `runs/`）。  

## 1. 变更范围（最小可评审）

### 1.1 代码
- `AMRBicycleEnv`（森林自行车环境）新增 progress 距离模式：
  - `euclid`（默认，旧行为）
  - `grid4`（v8p1）
  - `dijkstra8_nocorner`（v8p2）
- progress 距离采样改为 finite-safe，避免 `inf/NaN` 传导到 reward/gating/fallback。

### 1.2 CLI/配置
- `--forest-progress-dist-mode` choices 增加 `dijkstra8_nocorner`。
- 新增 costmap 参数：
  - `--forest-progress-cost-w-clearance`（clearance 惩罚权重，>=0）
  - `--forest-progress-cost-sigma-m`（衰减长度，>0，单位 m）

### 1.3 测试
- 增加单元测试覆盖 Dijkstra（对角/禁穿角/加权/不可达）与 finite-safe 插值稳定性。

### 1.4 文档与版本留档
- 新增 `configs/v8p2.json` + `configs/repro_20260223_v8p2_costmap_smoke.json`（可复现实验快照）。
- 新建版本四件套目录：`docs/versions/v8p2/`（中文）。
- 同步 `configs/INDEX.md`、根 `README.md` 与 `README.zh-CN.md`、`docs/versions/README.md` 的索引口径。

## 2. 实施步骤（3–7 步）

1) 写设计文档并提交（`docs/plans/2026-02-23-v8p2-costmap-dijkstra-design.md`）。  
2) 实现 `dijkstra8_nocorner` 距离场 + finite-safe 采样（env 侧），并补齐单测。  
3) CLI 透传新 mode 与 cost 参数（train/infer），本地跑 `pytest` + self-check。  
4) 新增 v8p2 profiles（`v8p2.json` + repro config），落版本四件套（先写 `N/A` 占位）。  
5) 远端 `ubuntu-zt` 跑 smoke（train `episodes=150` + infer `runs=3`），回传 `runs/`。  
6) 回填 v8p2 结果与 run 追溯，更新索引，提交并 `git push`。  

## 3. 风险点与兜底

1) Dijkstra 计算开销偏大：若 smoke wall-time 明显上升，优先优化（减少重复计算/缓存/向量化），再考虑降采样。  
2) cost 超参不稳导致保守/timeout：先在 smoke 上做小范围 sweep（`w_clearance` / `sigma_m`），以 SR 作为 hard gate。  
3) 不可达区域导致 `inf/NaN`：通过 finite-safe 插值保证 reward/gating 不被污染；必要时对不可达直接回退到欧氏 progress。  

## 4. 验证方式（最小闭环）

```bash
conda run -n ros2py310 python -m pytest -q
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

远端 smoke（示例）：
```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260223_v8p2_costmap_smoke"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260223_v8p2_costmap_smoke"
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p2_costmap_smoke/ runs/v8p2_costmap_smoke/
```


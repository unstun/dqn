# 仓库整理报告（2026-02-21）

## 1. 本轮目标

- 目标：降低 agent/人协作时的“口径分叉”和“归档遗漏”风险。
- 范围：仅整理文档与索引，不修改训练/推理代码与算法实现。

## 2. 已完成事项

1. **补齐 `v7p1` 版本归档四件套**
   - `docs/versions/v7p1/README.md`
   - `docs/versions/v7p1/CHANGES.md`
   - `docs/versions/v7p1/RESULTS.md`
   - `docs/versions/v7p1/runs/README.md`
2. **同步三处版本索引**
   - 根目录：`README.md`
   - 中文入口：`README.zh-CN.md`
   - 目录镜像：`docs/versions/README.md`
3. **README 口径可解析化（AI 优先）**
   - 顶部加入 `AI TL;DR` 合同块（`STABLE_PROFILE`、`SMOKE_GATE`、`FINAL_GATE`、`REPORT_ARTIFACTS`）。
   - 统一阶段术语：`self-check` / `micro-smoke` / `smoke` / `full`。
   - 明确 KPI 工件职责：`table2_kpis_mean_raw.csv`（结论）与 `table2_kpis_raw.csv`（诊断）。
4. **消除 `Claude.md` 规则歧义**
   - 将 `Claude.md` 改为“弃用跳转说明”，明确 canonical 为 `CLAUDE.md` + `AGENTS.md`。
5. **新增版本归档执行清单**
   - 新文件：`docs/versions/CHECKLIST.md`。

## 3. 当前状态判断（结构层面）

- 结论：**中度可维护，但尚未完全规范化**。
- 主要风险点：
  - 历史版本跨度大（`v1` 到 `v7p2p1`），早期口径与当前口径并存。
  - `smoke` 与 `micro-smoke` 在历史文档中仍有混写，需要持续治理。
  - 新实验若未按四件套与三处索引同步，容易再次出现“跑了但找不到结论入口”。

## 4. 下一步整理优先级（建议）

### P0（每轮必须做）

- 每次 train/infer 结束后，同轮更新对应 `docs/versions/<version>/` 四件套。
- 同步三处索引（`README.md`、`README.zh-CN.md`、`docs/versions/README.md`）。
- 使用 `docs/versions/CHECKLIST.md` 逐项勾选，避免遗漏。

### P1（1~2 轮内完成）

- 在各活跃版本（当前建议 `v7p1`、`v7p2`、`v7p2p1`）文档内统一阶段命名与门槛口径。
- 将“最终结论”页面只保留 `full` 证据（short/long 各 `runs=20`），其余标注为阶段性结果。

### P2（后续逐步）

- 对 `v1~v6` 历史文档做轻量口径补注（不改历史结论，只加“当时口径”说明）。
- 形成可脚本化的“索引一致性检查”（后续可再自动化）。

## 5. 本轮未做（明确边界）

- 未改动任何 `train.py` / `infer.py` / `forest_vehicle_dqn/*` 代码。
- 未新增 Python 依赖、未调整配置 schema。
- 未执行长时训练或 full 评测（仅文档整理）。

## 6. 建议的日常最小流程（5 分钟版）

1. 跑完实验后先记三条路径：`run_dir`、`run.json`、`table2_kpis_mean_raw.csv`。
2. 同步更新该版本四件套（至少填命令、路径、关键指标、`failure_reason`）。
3. 更新三处索引的同一行。
4. 执行：
   - `diff -u AGENTS.md CLAUDE.md`
   - `rg -n -- "STABLE_PROFILE|SMOKE_GATE|FINAL_GATE|REPORT_ARTIFACTS" README.md README.zh-CN.md`
5. 再进入下一轮迭代。

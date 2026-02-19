# dqn/AGENTS.md（Ubuntu 24.04 + ros2py310）

> **作用域**：本文件适用于 `/home/sun/phdproject/dqn/dqn/**`。仓库通用说明见 `../AGENTS.md`。若冲突，以本文件为准。
> **Scope**: applies to `/home/sun/phdproject/dqn/dqn/**`. Repo-wide notes: `../AGENTS.md`. If conflict, this file wins.

## 0. 总原则（必须遵守）/ Core rules (must follow)

1) **先计划后动手 / Plan before action**

   - 写/改任何 文件前：必须先输出 3–7 步「实施计划」+「将改动的文件清单」+「风险点」+「验证方式」。
   - 输出计划后：默认等待你明确确认（例如“开始/实现/按计划执行”）。如你在当次任务中明确允许“自检后直接开始”，则在完成约定自检并通过后开始实际修改。
   - 未进入实施前，仅允许做非侵入式探索（读文件、搜索、运行不改代码的检查/测试）。
   - Before changing any *repo-tracked* file: output a 3–7 step plan + files-to-change list + risks + verification.
   - After the plan: wait for explicit user confirmation by default. If the user explicitly allows “start after self-check” for that task, begin once the agreed self-check passes.
   - Before implementation, exploration-only is allowed.
2) **小步提交 & 可回滚 / Small, reversible changes**

   - 每次改动尽量小、单一目的；避免“顺手重构/顺手格式化全仓库”。
   - 涉及大量文件/大范围行为变化：拆成多个可独立 review 的步骤。
   - Keep each change small and single-purpose; avoid opportunistic refactors or repo-wide formatting.
   - Split large changes into independently reviewable steps.
3) **最小高置信变更 / Minimal high-confidence change**

   - 修 bug：优先 “加失败测试 → 修复 → 全绿”。
   - 重构：必须保证行为不变（见 DoD），并说明如何验证。
   - Bugfix: prefer “failing test → fix → green”.
   - Refactor: preserve behavior; state how to verify it.
4) **默认不引入新依赖 / No new deps by default**

   - 未经明确说明，不新增依赖/不升级大版本。确需新增：先解释理由、替代方案、影响面。
   - Do not add dependencies or major upgrades unless explicitly approved; justify necessity and alternatives.
5) **称呼约定 / Addressing convention**

   - 每次回复默认以“`帅哥，`”开头（除非你明确要求不需要）。
   - Start responses with “`帅哥，`” by default unless the user asks otherwise.

5.1) **沟通口吻 & 标识符释义 / Communication tone & identifier glossing**

   - 默认用“研究生汇报”口吻：先说清楚我对问题的理解/假设与不确定性，再给可执行的步骤与验证方式；避免营销式/口号式表达。
   - 文本中首次出现不直观标识符（函数/类/变量/参数/CLI 选项/文件名等）时，在其后用括号补一句短解释（不超过 1 句）；重复出现可省略。
   - 示例：`epsilon`（ε-greedy 的探索率/随机动作概率）、`gamma`（折扣因子）、`rollout_agent(...)`（用当前策略在环境里采样轨迹/回合）、`--runs 20`（评测重复次数）。
   - Default to a "grad student report" tone: state understanding/assumptions/uncertainties first, then actionable steps + verification; avoid marketing/slogan-style wording.
   - On first mention of a non-obvious identifier (function/class/variable/arg/CLI flag/path, etc.), add a short parenthetical gloss (≤1 sentence); omit on repeats.
   - Examples: `epsilon` (epsilon-greedy exploration rate / random-action probability), `gamma` (discount factor), `rollout_agent(...)` (sample trajectories/episodes in the env with the current policy), `--runs 20` (number of evaluation repeats).
6) **可复现性（config）/ Reproducibility (configs)**

   - 默认规则：每次完成任何**代码改动**（含重构/性能优化/数值变更）后，都要新增一份“可复现 config”，保证你本人可复现（你可在当次任务中明确豁免）。纯文档改动默认不强制。
   - 推荐做法：在 `configs/` 新增 `repro_YYYYMMDD_<topic>.json`，并在其中记录：
     - 复现实验/自检命令（推荐 `conda run -n ros2py310 ...`）
     - 关键超参/seed、输入数据/地图、评估指标口径
     - 变更点摘要（可用 `_meta` 字段，避免被严格解析为 CLI 参数）
   - Default: after any **code change** (including refactors/perf tweaks/numerical changes), add a new reproducibility config (can be waived per-task by the user). Docs-only changes are optional.
   - Prefer `configs/repro_YYYYMMDD_<topic>.json` and record commands/hyperparams/metrics + a `_meta` summary.
7) **环境约束 / Environment**

   - 环境默认为 Ubuntu 24.04；Python 默认使用 conda env：`ros2py310`。
   - Default environment: Ubuntu 24.04; Python via conda env `ros2py310`.
8) **仓库研究目标 / Research north-star**

   - 最终目标：创新一个改进的深度强化学习运动规划方法，并与多类 baseline 对比（普通强化学习、以及传统算法+MPC）。
   - 当前思路：使用 CNN 表征（或 CNN+其他结构），并在 DDQN 上做可验证、可复现的改进；后续可变，但必须有实验支撑。
   - Goal: propose an improved deep-RL path-planning method and compare against multiple baselines (standard RL, classical planners, and classical+MPC).
   - Current direction: CNN representations + verifiable DDQN improvements; must be supported by reproducible experiments.
9) **学术定义合规（硬约束）/ Academic-definition compliance (hard requirement)**

   - 任何算法名/术语（如 DQN/DDQN/Double Q-learning、CNN 架构、Hybrid A*/RRT*、MPC 等）必须满足其学术/原论文定义；禁止“只改风格但不符合定义”的实现。
   - 若对定义/标准模糊：优先下载/整理对应论文 PDF 或官方实现仓库作为参考，并在说明中引用；建议放入 `paper/`（或记录可追溯链接/出处）。 github仓库可以单独建立文件夹
   - Any claimed algorithm/term must match the academic definition; do not label something “DDQN” unless it is DDQN.
   - If unclear: retrieve the paper / reference implementation and cite it; prefer storing PDFs under `paper/`.
10) **规则可追加 / Rules may evolve**

- 以上规则可随时追加；新增规则以最新说明为准。
- Rules can be extended; the newest rules take precedence.

11) **README 同步（训练/推理命令）/ README sync (train/infer commands)**

- 每次涉及代码或配置行为的改动后，默认同步更新 `README.md` 与 `README.zh-CN.md` 中“最新训练/推理命令”（含对应 profile 名）。
- 若命令未变化，也应明确检查并保持中英文两份 README 一致，避免漂移。
- After any code/config behavior change, update the "latest train/infer commands" in both `README.md` and `README.zh-CN.md` (including profile names).
- If commands are unchanged, still verify and keep both READMEs aligned to avoid drift.

12) **时间优先验证流程 / Time-first validation pipeline**

- 深度强化学习迭代默认采用两阶段流程：`self-check -> smoke -> full runs=20`。
- 未经明确说明，每轮先完成 smoke（短训练+短评测）再决定是否进入 full 评测，避免直接长时全量实验。
- 推荐 smoke 命令使用 `conda run -n ros2py310 ...`，并限制在可快速回路内完成。
- Use a two-stage loop by default: `self-check -> smoke -> full runs=20`; do not jump to long full evaluations unless smoke indicates clear progress.

13) **最终研究门槛（硬约束）/ Final research gate (hard requirement)**

- 最终结论必须在 `short/long` 双套件、各 `runs=20` 条件下汇报。
- 对标 `Hybrid A*-MPC` 时，`CNN-DDQN` 至少满足：
  - `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
  - `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
  - `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`
- 若任一套件未满足上述三条，视为未通过最终门槛。
- Final claims require short/long suites with `runs=20` each, and the above three inequalities against `Hybrid A*-MPC`.

14) **推理期策略口径（命名必须一致）/ Inference policy regimes (naming must match implementation)**

- 本仓库允许在**推理阶段**使用 `admissible-action masking` / 安全过滤 / `top-k` 重选 / `stop override` / replacement / fallback / planner(如 A*/MPC) takeover 等混合策略，以提升成功率与安全性。
- 但必须显式区分并正确命名（论文/文档/版本留档必须与实现一致）：
  - `strict-argmax`（等价于旧口径 strict no-fallback）：推理阶段必须使用纯 `argmax(Q)` 直接出动作；允许计算 mask 仅用于统计/诊断，但不得影响最终动作。
  - `shielded` / `masked` / `hybrid`：推理期允许上述干预（masking/top-k/override/replacement/fallback/takeover 等）；此时不得宣称 `strict-argmax` / strict no-fallback。
- 训练期仍可使用合法数据筛选/课程/辅助损失；若推理期启用干预，必须在 config 与版本四件套中写清楚具体策略与开关。

- This repo allows inference-time hybrid policies (e.g., admissible-action masking / safety filtering / top-k retry / stop override / replacement / fallback / planner takeover) to improve safety and success rate.
- You MUST name the regime honestly and keep docs/papers/archives consistent with the implementation:
  - `strict-argmax` (same as the legacy "strict no-fallback" label): inference must be pure `argmax(Q)`; masks may be computed for logging/diagnostics only, but must NOT affect the executed action.
  - `shielded` / `masked` / `hybrid`: inference-time interventions are enabled; you must NOT claim `strict-argmax` / strict no-fallback in this case.

15) **版本命名与留档（硬约束）/ Versioning & logging (hard requirement)**

- 版本命名统一采用 `vxpx`：**大改动**执行 `v+1`；**小改动**执行 `p+1`。
- 每个版本必须使用独立文件夹：`docs/versions/<version>/`。
- 每版至少包含：`README.md`（版本总结）、`CHANGES.md`（具体改动）、`RESULTS.md`（结果对比）、`runs/README.md`（对应 run 路径与口径）。
- 每个版本必须写 MD 留档（方法、详细的修改的地方、参数、命令、结果、结论、下一步）。
- 推荐在新一轮开始前先读取上一版本留档，防止重复试验与口径漂移。

15.1) **版本更新前 GitHub 快照（硬约束）/ GitHub snapshot before version bump (hard requirement)**

- 适用范围：任何准备发布新版本（`vxpx`）的代码/配置改动，在**进入实施**之前。
- 开始实施前必须确保 `git status`（工作区状态）为 clean；若不 clean，先 `git add/commit`（提交本地改动形成可回退点）或 `git stash`（临时存放改动）处理到 clean。
- 必须执行 `git push`（推送当前分支到远端 `origin`）并确认成功后，才允许开始改代码/改配置/创建新版本目录。
- 推荐：同时打快照 tag（`git tag -a <version>-pre -m "pre-change snapshot"`）并 `git push --tags`（推送标签），便于一键回退。
- 若 `git push`/`git push --tags` 失败：立即停止实施，先排查 SSH/权限/网络/远端分支保护等问题，确保远端有可回退快照。

16) **版本归档标准（执行细则，硬约束）/ Archive standard (operational, hard requirement)**

- 适用范围：`v1` 起全部版本（含历史整理与后续新增）。
- 目录结构必须是：`docs/versions/<version>/`；禁止仅保留单文件 `docs/versions/<version>.md` 作为正式留档。
- 每个版本目录必须包含且仅少不了以下四件套：
  - `README.md`：版本目标、方法摘要、关键命令、代表 run、结论、下一步。
  - `CHANGES.md`：相对上一版的代码/配置改动明细（推荐 `old -> new` 口径）与受影响文件清单。
  - `RESULTS.md`：`table2_kpis_mean_raw.csv` 来源路径、short/long 指标、基线对比、门槛检查、failure_reason 分布。
  - `runs/README.md`：代表 run 的 `run_dir/run_json/kpi` 路径，以及该版本可追溯 run 列表。
- 归档入口统一为仓库根目录 `README.md` 中“版本总索引（v1 → ...）”；`docs/versions/README.md` 作为子目录镜像索引同步维护。
- 指标与路径必须可追溯到真实文件；禁止编造历史结果。无法确认的数据必须写 `N/A` 并注明原因。
- `docs/versions/README.md` 必须同步更新（版本号、目录路径、short/long 最佳 SR、baseline、状态）。
- 历史版本整理时，允许保持算法不变，仅做文档结构与口径标准化；不得借机改写实验结论。
- 归档语言要求：`docs/versions/**` 下版本留档四件套默认必须使用中文撰写（含 `README.md`、`CHANGES.md`、`RESULTS.md`、`runs/README.md`）。
- 仅当你明确要求时，才允许对应版本留档使用英文或中英双语。
- 为保证可执行与可复现，命令行、文件路径、参数名、代码标识符可保留英文原文，不要求翻译。
- 每次运行完都要求归档：任何一次运行（含 `self-check`、`smoke`、`full`、训练命令、推理命令）结束后，必须在同一工作轮次内更新对应版本留档。
- 最小归档内容：在对应 `docs/versions/<version>/` 四件套中至少记录本次运行命令、`run_dir`、`run_json`、`kpi` 路径、short/long 关键指标与 `failure_reason` 分布（若该次运行产出该字段）。
- 运行失败也必须归档：若结果缺失，必须写 `N/A` 并注明失败原因（如报错中断、超时、人工终止）。

17) **v9+ 追加流程（硬约束）/ v9+ append workflow (hard requirement)**

- 新版本（`v9` 及后续 `vxpx`）创建时，必须在同一轮提交内同时完成三处更新：
  1. 新建 `docs/versions/<version>/` 四件套。
  2. 更新根目录 `README.md` 的“版本总索引”。
  3. 更新 `docs/versions/README.md` 镜像索引。
- 版本索引必须显式记录：版本号、目录、主 config、关键 run、short/long 最佳 SR、基线 SR、状态。
- `config ↔ runs` 映射必须锁定到真实路径；若该版本仅有 baseline-only（`--skip-rl`）输出，必须在 `RESULTS.md` 与 `runs/README.md` 明确标注，并将 RL 指标写为 `N/A`。
- 若不存在独立 config（仅有 `--profile ... --out ...` 变体运行），必须在版本留档说明“继承 profile + argv 覆盖”并引用对应 `run.json`。

## 默认环境 / Default environment

- OS: Ubuntu 24.04
- Conda env: `ros2py310`
- 命令风格 / Command style: 优先 / prefer `conda run -n ros2py310 ...`
- 安装/快速开始/成功判定：优先参考 `../AGENTS.md`（避免重复维护）。
  Install/quickstart/success definition: refer to `../AGENTS.md`.

## 最小自检 / Minimal self-check

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

## DoD / 验收标准（最小）

- 文档层面 / Docs
  - `AGENTS.md` 结构清晰、无明显歧义；与 `../AGENTS.md` 不冲突（重复信息尽量引用上层）。
  - 引用的关键路径存在：`train.py`、`infer.py`、`configs/`、`paper/`。
- 过程层面 / Process
  - 后续任何实现改动均遵守：先计划后动手、小步变更、默认不引入依赖、学术定义合规、可复现性约束（除非你明确豁免）。
  - 默认执行两阶段验证（smoke 优先），最终结论使用 short/long + runs=20 的硬门槛口径。

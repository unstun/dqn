# `configs/` 索引（2026-02-24）

> 目标：把“该用哪个 profile（配置名）”先讲清楚，避免从 100+ 文件里猜。

## 1) 主线/优先使用

| 类别 | 文件 | 用途 | 状态 |
|---|---|---|---|
| 稳定主线 | `configs/v7p1.json` | 当前训练/推理默认主线 | 推荐 |
| V8 迭代入口（候选） | `configs/v8p12.json` | 口径对齐 shortest-path progress-dist（`w_clearance=0`），主攻 long detour（目标：C 门槛） | smoke 已跑（fixed pairs3，runs=3）：short 明显回退；long detour 仅小幅回落但仍落后 baseline → NO-GO（full 暂不建议） |
| V8 迭代（上一候选） | `configs/v8p11.json` | 训练侧强化（expert+D QfD 更强，继承 v8p10 推理口径；目标：C 门槛） | smoke 已跑（short 打败 baseline；long 仍落后）→ full gate 暂不建议 |
| V8 迭代（上一候选） | `configs/v8p10.json` | progress-dist clearance 消融（`dijkstra8_nocorner` + `progress_cost_w_clearance` sweep；目标：C 门槛） | infer-only sweep smoke 已跑（fixed pairs3，runs=3；当前 best：w=2.0；baseline gap）→ full gate 暂不建议 |
| V8 迭代（上一候选） | `configs/v8p9.json` | infer-sweep（SR≈1.0 前提下压 `avg_path_length/path_time_s`；目标：C 门槛） | sweep smoke 已跑（fixed pairs3，runs=3）；full gate C 暂不建议（baseline gap） |
| V8 迭代（上一候选） | `configs/v8p8.json` | dueling + globalcnn_fusion + aux admissibility（更强表征 + 可行性辅助监督，目标：C 门槛） | smoke 已跑（NO-GO；mid/long 路径与时间劣于 baseline）→ full gate 待跑 |
| V8 迭代（上版） | `configs/v8p7.json` | goal-approach speed shaping（接近目标阶段速度整形：推理侧更早减速，避免末段“必撞态”） | infer-only smoke 通过（fixed v8p6 checkpoint；SR=1.0）；train+infer smoke 待跑 |
| V8 迭代（上版） | `configs/v8p6.json` | replace-topq（替换候选 Top-Q 约束：把 tie-break 限制在高 Q 小集合内） | infer-only smoke 通过（topq=1/2/3）；train+infer smoke NO-GO（short/mid collision=1/3） |
| V8 迭代（上版） | `configs/v8p5.json` | replace-ranking 消融（argmax 不可行时的替换动作排序：Q vs progress/clearance tie-break） | 回归通过（fixed pairs）；infer-only：tie-break short 有 collision |
| V8 迭代（上版） | `configs/v8p4.json` | short-rollout fallback 的 1-step collision-free 降阶兜底（避免“最后兜底选到立即碰撞动作”） | 回归 FAIL（collision+timeout）；暂不 smoke |
| V8 迭代（上上版） | `configs/v8p3.json` | collision-first fallback safety（避免 `min_od_m` 筛空导致落入碰撞兜底） | smoke 已跑（mid collision，long timeout） |
| V8 迭代（上上上版） | `configs/v8p2.json` | costmap Dijkstra progress distance（`forest_progress_dist_mode=dijkstra8_nocorner`） | smoke 已跑（short 有 collision） |
| 失败分支归档 | `configs/v8p1.json` | `v8p1` navdist progress distance（`forest_progress_dist_mode=grid4`） | 仅复盘 |
| 失败分支归档 | `configs/v7p3p4.json` | `v7p3p4` safe fallback 补丁（progress mask 为空时回退 safe 动作集，修复 `fallback_rate`） | 仅复盘 |
| 失败分支归档 | `configs/v7p3p3.json` | `v7p3p3` 调参（`tp=0.3` + `min_prog=0.0`）尝试降低 timeout | 仅复盘 |
| 失败分支归档 | `configs/v7p3p2.json` | `v7p3p2` turn-aware top-k 抑制急拐方案 | 仅复盘 |
| 失败分支归档 | `configs/v7p3p1.json` | `v7p3p1` 通用自适应 no-progress 惩罚方案 | 仅复盘 |
| 失败分支归档 | `configs/v7p3.json` | `v7p3` short/long 分离 no-progress 惩罚方案 | 仅复盘 |
| 失败分支归档 | `configs/v7p2.json` | `v7p2` 试验分支留档 | 仅复盘 |
| 回退对照 | `configs/v6p2p3.json` | 主线前一稳定代口径对照 | 仅对照 |

## 2) 版本链配置（`v*.json`）

这些文件是版本号入口（`vxpx` 命名），建议优先用于版本复现实验：

- `configs/v6p2.json`
- `configs/v6p2p1.json`
- `configs/v6p2p2.json`
- `configs/v6p2p3.json`
- `configs/v7p1.json`
- `configs/v7p2.json`
- `configs/v7p3.json`
- `configs/v7p3p1.json`
- `configs/v7p3p2.json`
- `configs/v7p3p3.json`
- `configs/v7p3p4.json`
- `configs/v8p1.json`
- `configs/v8p2.json`
- `configs/v8p3.json`
- `configs/v8p4.json`
- `configs/v8p5.json`
- `configs/v8p6.json`
- `configs/v8p7.json`
- `configs/v8p8.json`
- `configs/v8p9.json`
- `configs/v8p10.json`
- `configs/v8p11.json`
- `configs/v8p12.json`

> 规则：新增版本时，优先新增 `v*.json`，再在四件套中记录它与 `run_dir`（运行目录）的映射。

## 3) 历史别名配置（`forest_*.json`）

`forest_*`（早期语义化别名）保留用于兼容旧命令，不建议作为新版本主入口：

- `configs/forest_a_all6_300_cuda.json`
- `configs/forest_a_all6_300_cuda_latest.json`
- `configs/forest_a_all6_300_cuda_latest_final.json`
- `configs/forest_a_all6_300_cuda_latest_final_no_stuck.json`
- `configs/forest_a_best.json`
- `configs/forest_a_best_600_cuda.json`
- `configs/forest_a_best_latest.json`
- `configs/forest_a_best_latest_cuda.json`
- `configs/forest_a_best_two_suites.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck_long_baselines_100pct.json`
- `configs/forest_a_cnn_ddqn_300_cuda_latest_final_no_stuck_long_baselines_tuned.json`

## 4) 实验快照配置（`repro_*.json`）

- 数量较多（当前 90+），用于“一次实验一份参数快照”的可追溯记录。
- 建议做法：新实验继续写 `repro_YYYYMMDD_<topic>.json`，但最终沉淀到版本链时再抽取 `v*.json` 作为主入口。

## 5) pairs 文件（固定样本集）

`pairs`（固定随机起终点集合）用于公平对比，不是 profile：

- 示例：`configs/repro_20260210_forest_a_pairs_short20_v1.json`
- 示例：`configs/repro_20260210_forest_a_pairs_long20_v1.json`
- 调用方式：`infer.py --rand-pairs-json <path>`

## 6) 建议命名与落库规则（从本轮开始）

1. 主线版本：`v*.json`（长期入口）。
2. 过程实验：`repro_YYYYMMDD_<topic>.json`（短期快照）。
3. 禁止在新流程中继续扩增 `forest_*_latest*` 类别。
4. 每个新 `v*.json` 都必须在 `docs/versions/<version>/` 四件套登记：
   - 配置路径
   - 代表命令
   - 代表 `run_dir`
   - `table2_kpis_mean_raw.csv` 与 `table2_kpis_raw.csv` 路径

## 7) 最小使用模板

```bash
conda run -n ros2py310 python train.py --profile v7p1
conda run -n ros2py310 python infer.py --profile v7p1
```

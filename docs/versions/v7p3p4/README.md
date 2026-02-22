# v7p3p4 版本说明

- 版本类型：**Patch（p+1）**
- 上一版本：`v7p3p3`（失败归档）
- 稳定对照基线：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**已运行（infer smoke：runs=3，固定 `v7p3p2` checkpoint），失败归档；主线保持 `v7p1`**

## 本版目标
- 修复推理期 admissible gating 的兜底：当 `argmax(Q)` 不可采纳且“进度可采纳动作集为空”时，不再回退到原始 `argmax(Q)`，而是回退到 collision-safe 动作集合，避免 `collision` 回潮并尽量恢复 `success_rate`。

## 方法摘要（代码补丁：safe fallback）
- 推理动作选择（`rollout_agent(...)`（用当前策略在环境里采样轨迹/回合））：
  - 先尝试 top-k 的 admissible replacement；
  - 若无可用替换，再尝试 progress-mask；
  - **若 progress-mask 为空，则 `fallback_to_safe=True` 回退到安全动作集**；
  - 若仍为空，则使用 `_fallback_action_short_rollout(...)`（短视安全回退动作选择）。
- 训练侧 greedy 评估动作选择（`_forest_policy_action_from_q(...)`（从 Q 值计算 forest 推理动作））同步同一套兜底逻辑，保证训练/评估/推理行为一致。
- 统计修复：`fallback_rate`（推理期动作替换比例）现在计入 “最终动作 != 原始 `argmax(Q)`” 的步数，避免长期恒为 0。

## 本轮关键命令（实际执行）
- 本地 -> 远端同步（不含 `runs/`）：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 infer smoke（固定模型，不重训）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p4_safe_fallback_infer_smoke"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p4_safe_fallback_infer_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p4_safe_fallback_infer_smoke/`

## 固定模型来源（本轮不重训）
- `models_run_dir`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744`

## 代表 run
- 推理：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513`
- KPI（均值）：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_mean_raw.csv`
- KPI（逐回合）：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_raw.csv`

## 核心结果摘要（runs=3）

| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | argmax_inadmissible_rate | fallback_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.667 | 22.7499 | 13.2250 | 0.164133 | 0.192 | 0.192 |
| short | Hybrid A*-MPC | 1.000 | 17.0342 | 10.2667 | 0.114272 | N/A | N/A |
| mid | CNN-DDQN | 0.667 | 36.8081 | 20.7000 | 0.190047 | 0.470 | 0.470 |
| mid | Hybrid A*-MPC | 1.000 | 24.0814 | 13.3333 | 0.058072 | N/A | N/A |
| long | CNN-DDQN | 1.000 | 71.4983 | 38.1000 | 0.171134 | 0.327 | 0.327 |
| long | Hybrid A*-MPC | 1.000 | 43.0107 | 22.8167 | 0.068718 | N/A | N/A |

## `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=2`，`timeout=1`
- long：`reached=3`
- 合计：`reached=7`，`timeout=2`（`collision=0`）

## 结论与下一步
- 本轮 smoke 结论：**NO-GO**（short/mid `success_rate` 仍低于 baseline，且 `path/time/smoothness` 明显劣于 baseline）。
- 但相对 `v7p3p3` 的关键进展：
  - short/mid 的 `collision` 现象消失，SR 回到 `v7p3p2` 推理消融的均衡点（`0.667/0.667/1.000`）；
  - `fallback_rate` 指标可用，可直接量化“策略对干预的依赖程度”。
- 下一版建议（`v7p3p5`）：在保持 safe fallback 的前提下，优先做“让策略本身更少触发 inadmissible/干预”的训练侧改动（例如引入 `q_margin`（Q 差距阈值）限制替换触发、增加更强的专家混入/模仿约束、或重新对齐 reward 以压缩 path/time）。  

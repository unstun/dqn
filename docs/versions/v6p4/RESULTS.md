# v6p4 结果

## 数据来源
- 主配置：`configs/v6p4.json`
- 复现配置：`configs/repro_20260220_v6p4_cnn_ddqn_schedule_demo_balance.json`
- 训练 run：`N/A`（待运行）
- 推理 run：`N/A`（待运行）
- KPI 文件：`N/A`（待运行）

## 一、本轮执行结论
- 本轮已完成 `v6p4` 配置升级与版本留档。
- `2026-02-20` 已完成 `train.py --profile v6p4 --self-check` 与 `infer.py --profile v6p4 --self-check`，两者均通过并识别 `cuda:0`。
- `2026-02-20` 已完成 `eps_decay` 自适应改动后的 self-check 复验，结果仍为通过（训练/推理入口与 CUDA 检查均正常）。
- 尚未执行 smoke/full，因此本文件当前仅记录配置变更与 self-check 结果，不做效果结论。

## 二、指标总表（short/mid/long）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | planning_time_s | argmax_inadmissible_rate | failure_reason |
|---|---|---:|---:|---:|---:|---:|---|
| short | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| short | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |
| mid | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| mid | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |
| long | CNN-DDQN | N/A | N/A | N/A | N/A | N/A | N/A |
| long | Hybrid A*-MPC | N/A | N/A | N/A | N/A | N/A | N/A |

## 三、failure_reason 汇总
- `CNN-DDQN`：`N/A`
- `Hybrid A*-MPC`：`N/A`

## 四、门槛检查（short/long + runs=20）
- `N/A`（本轮未执行 full `runs=20`）

## 五、待补动作
1. 执行 smoke（推荐 `episodes=300, runs=3`）并回填 KPI 与 `failure_reason` 分布。
2. smoke 通过后执行 full（`runs=20`）并做最终门槛判定。
3. 将 full 的 short/long 门槛检查结论回填到本文件与版本索引。

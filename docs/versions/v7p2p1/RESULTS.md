# v7p2p1 结果（失败归档）

## 数据来源
- 训练 run：`runs/v7p2_es150/train_20260220_222056`
- 推理 run（runs=20）：`runs/v7p2_es150/train_20260220_222056/infer/20260220_223016`
- 推理 run（runs=3）：`runs/v7p2_es150_eval3/20260220_223301`
- 训练评估（150 轮对照）：
  - `runs/v7p2_es150/train_20260220_222056/training_eval.csv`
  - `runs/v7p1_remote150/train_20260220_182740/training_eval.csv`
- 对照失败记录（v7p1 runs=20 无法直接复跑）：见 `docs/versions/v7p2p1/runs/README.md`

## 一、v7p2_es150 推理指标（runs=20）
| 套件 | 算法 | success_rate | avg_path_length | path_time_s | avg_curvature_1_m | planning_time_s |
|---|---|---:|---:|---:|---:|---:|
| short | CNN-DDQN | 0.85 | 23.3004 | 16.1235 | 0.163614 | 0.18829 |
| short | Hybrid A*-MPC | 0.95 | 15.7667 | 9.2526 | 0.084289 | 0.29049 |
| mid | CNN-DDQN | 0.80 | 28.7948 | 18.0438 | 0.150047 | 0.17626 |
| mid | Hybrid A*-MPC | 0.90 | 21.8143 | 12.2028 | 0.058645 | 0.17352 |
| long | CNN-DDQN | 0.65 | 66.0207 | 41.4731 | 0.226982 | 0.44093 |
| long | Hybrid A*-MPC | 1.00 | 42.9293 | 22.7625 | 0.059527 | 1.41134 |

## 二、failure_reason 分布（runs=20）
- CNN-DDQN：
  - short：`reached=17`, `collision=3`
  - mid：`reached=16`, `collision=4`
  - long：`reached=13`, `collision=3`, `timeout=4`
- Hybrid A*-MPC：
  - short：`reached=19`, `collision=1`
  - mid：`reached=18`, `collision=2`
  - long：`reached=20`

## 三、smoke 口径（runs=3）与历史对照
- `v7p2_es150_eval3`（`runs=3`）：
  - short：`sr=1.00`, `len=17.9152`, `time=11.1333`
  - mid：`sr=0.333`, `len=23.9996`, `time=13.3000`
  - long：`sr=1.00`, `len=66.7665`, `time=44.7167`
- `v7p1_smoke_eval`（历史 `runs=3`）：
  - short：`sr=1.00`, `len=16.2469`, `time=10.1333`
  - mid：`sr=0.667`, `len=26.0729`, `time=17.8250`
  - long：`sr=1.00`, `len=46.2812`, `time=30.4000`

## 四、150 轮训练内评估对照（ep=150）
- `v7p1_remote150`：`sr_short/sr_long/sr_all = 1.0 / 0.6 / 0.8`
- `v7p2_es150`：`sr_short/sr_long/sr_all = 0.8 / 0.6 / 0.7`

## 五、结论
- 在本轮标准流程下，`v7p2` 改动未体现稳定收益，且在 long 套件退化明显。
- 将本次尝试定义为失败版本 `v7p2p1`，主线回退到 `v7p1`。
- `v7p2p2` 将在 `v7p1` 基线上继续单变量迭代。

## 六、回退后可用性抽检
- 回退后已完成本地抽检推理（`v7p1_remote150`，short，runs=1），流程可正常执行。
- 抽检路径：`runs/rollback_v7p1_check/20260220_045505`

# v6p5 runs 追溯

## 1. 本轮命令
- self-check（已执行）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p5 --self-check`
- 尝试 1（中断，耗时过长）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 300 --out v6p5_bn_smoke300`
- 尝试 2（中断，耗时过长）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 40 --out v6p5_bn_smoke40`
- 尝试 3（失败，参数不满足环境约束）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 20 --max-steps 200 --train-eval-every 0 --no-forest-demo-prefill --forest-demo-pretrain-steps 0 --out v6p5_bn_quick20 --save-ckpt final`
  - 失败原因：`max_steps=200` 小于 `forest_a` 最小需求（报错要求至少 `511`）
- 尝试 4（成功，quick20）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 20 --max-steps 600 --train-eval-every 0 --no-forest-demo-prefill --forest-demo-pretrain-steps 0 --out v6p5_bn_quick20 --save-ckpt final`
  - `conda run -n ros2py310 python infer.py --profile v6p5 --models v6p5_bn_quick20 --runs 3 --out v6p5_bn_quick20`
- 尝试 5（标准 smoke，人工中断）：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 300 --out v6p5_bn_smoke300 --device cuda`
  - 状态：运行约 37 分钟后人工中断（终端中止），未产出模型与 KPI

## 2. run 路径登记
- 中断 run（300ep）：
  - train `run_dir`：`runs/v6p5_bn_smoke300/train_20260219_164257`
  - train `run.json`：`runs/v6p5_bn_smoke300/train_20260219_164257/configs/run.json`
  - 状态：`N/A`（人工中断，无模型与 KPI）
- 中断 run（40ep）：
  - train `run_dir`：`runs/v6p5_bn_smoke40/train_20260219_165120`
  - train `run.json`：`runs/v6p5_bn_smoke40/train_20260219_165120/configs/run.json`
  - 状态：`N/A`（人工中断，无模型与 KPI）
- 参数失败 run（quick20-错误参数）：
  - train `run_dir`：`runs/v6p5_bn_quick20/train_20260219_165717`
  - train `run.json`：`runs/v6p5_bn_quick20/train_20260219_165717/configs/run.json`
  - 状态：`N/A`（`max_steps` 约束报错退出）
- 有效 quick20 run：
  - train `run_dir`：`runs/v6p5_bn_quick20/train_20260219_165821`
  - train `run.json`：`runs/v6p5_bn_quick20/train_20260219_165821/configs/run.json`
  - train KPI：`runs/v6p5_bn_quick20/train_20260219_165821/training_returns.csv`
  - infer `run_dir`：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130`
  - infer `run.json`：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130/configs/run.json`
  - KPI（均值）：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130/table2_kpis_mean_raw.csv`
  - KPI（逐回合）：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130/table2_kpis_raw.csv`
- 标准 smoke 中断 run（cuda）：
  - train `run_dir`：`runs/v6p5_bn_smoke300/train_20260219_170903`
  - train `run.json`：`runs/v6p5_bn_smoke300/train_20260219_170903/configs/run.json`
  - 状态：`N/A`（人工中断，无模型与 KPI）

## 3. 结果登记
- self-check：
  - `train.py --profile v6p5 --self-check`：通过（`device_ok=cuda:0`）
  - `infer.py --profile v6p5 --self-check`：通过（`cuda_device_sm=sm_86`, `cuda_arch_supported=True`）
- quick20（有效）概要：
  - short：`CNN-DDQN SR=1.0`，`Hybrid A*-MPC SR=1.0`
  - mid：`CNN-DDQN SR=0.333`，`Hybrid A*-MPC SR=1.0`
  - long：`CNN-DDQN SR=0.0`，`Hybrid A*-MPC SR=1.0`

## 4. 备注
- 本版本核心变化为 `local` 观测口径 + `BatchNorm2d` 卷积归一化；旧 checkpoint 不兼容，需重训。
- `quick20` 为时间优先先验试跑（含 override），不替代标准 smoke/full 结论。
- 每次运行后需在同轮更新本文件与 `RESULTS.md`。

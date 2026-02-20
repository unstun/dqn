# v6p4 runs 追溯

## 1. 本轮命令
- self-check（已执行）：
  - `conda run -n ros2py310 python train.py --profile v6p4 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --self-check`
- smoke（待执行）：
  - `conda run -n ros2py310 python train.py --profile v6p4 --episodes 300 --out v6p4_smoke300`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --models v6p4_smoke300 --runs 3 --out v6p4_smoke300`
- full（待执行）：
  - `conda run -n ros2py310 python train.py --profile v6p4 --out v6p4_full3000`
  - `conda run -n ros2py310 python infer.py --profile v6p4 --models v6p4_full3000 --runs 20 --out v6p4_full3000`

## 2. run 路径登记
- train `run_dir`：`N/A`
- infer `run_dir`：`N/A`
- train `run.json`：`N/A`
- infer `run.json`：`N/A`
- KPI（均值）：`N/A`
- KPI（逐回合）：`N/A`

## 3. 结果登记
- self-check：
  - `train.py --profile v6p4 --self-check`：通过（`device_ok=cuda:0`）
  - `infer.py --profile v6p4 --self-check`：通过（`cuda_device_sm=sm_86`, `cuda_arch_supported=True`）
- smoke：`N/A`
- full：`N/A`

## 4. 备注
- 本版本继续保持 RL 仅 `cnn-ddqn`、基线仅 `Hybrid A*-MPC`。
- `2026-02-20` 已在 `eps_decay` 自适应改动后再次执行同一组 self-check，结果通过。
- 本文件需在每次运行后同轮更新，失败任务也必须写 `N/A + 原因`。

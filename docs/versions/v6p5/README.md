# v6p5 版本说明

- 版本类型：**Minor（p+1）**
- 上一版本：`v6p4`
- 本版口径：`shielded/hybrid`（推理阶段允许 `top-k`（Q 值前 k 个动作重选）与 `stop override`（到达目标位置后优先停稳动作））
- 状态：**已完成局部观测改造 + CNN BatchNorm 改造，已完成 quick20 先验试跑，待标准 smoke/full 评测**

## 本版目标
- 将 `obs_map_mode`（观测模式）从 `global`（整图下采样）切换为 `local`（局部裁剪后下采样）。
- 使用 `obs_local_range_m`（局部窗口半径，米）配置接近全图的 `18.0m` 覆盖范围，保持你要求的 mid360 近全图口径。
- 将 `obs_map_size`（观测地图边长）提升到 `64`，提高 CNN 输入分辨率。

## 方法概要
- `AMRBicycleEnv`（自行车模型环境）的 `_observe()`（观测构造函数）新增局部观测分支：每步以车体位置为中心裁剪 occupancy patch，再缩放到 `obs_map_size`。
- `train.py` 与 `infer.py` 新增并透传参数：`--obs-map-mode`、`--obs-local-range-m`。
- `CNNQNetwork`（CNN 版 Q 网络）卷积骨干引入 `BatchNorm2d`（二维批归一化层），结构为 `Conv2d -> BatchNorm2d -> ReLU`。
- 其余训练主线保持与 `v6p4` 一致（`cnn-ddqn`、`eps_decay` 自适应、DQfD 强度设置等）。

## 关键命令
- 自检：
  - `conda run -n ros2py310 python train.py --profile v6p5 --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p5 --self-check`
- smoke：
  - `conda run -n ros2py310 python train.py --profile v6p5 --episodes 300 --out v6p5_smoke300`
  - `conda run -n ros2py310 python infer.py --profile v6p5 --models v6p5_smoke300 --runs 3 --out v6p5_smoke300`
- full：
  - `conda run -n ros2py310 python train.py --profile v6p5 --out v6p5_full3000`
  - `conda run -n ros2py310 python infer.py --profile v6p5 --models v6p5_full3000 --runs 20 --out v6p5_full3000`

### CUDA 优先 profile（吞吐优先）
- 配置：`configs/v6p5_cuda_pref.json`
- 自检：
  - `conda run -n ros2py310 python train.py --profile v6p5_cuda_pref --self-check`
  - `conda run -n ros2py310 python infer.py --profile v6p5_cuda_pref --self-check`
- smoke：
  - `conda run -n ros2py310 python train.py --profile v6p5_cuda_pref --device cuda --episodes 300 --out v6p5_cuda_pref_smoke300`
  - `conda run -n ros2py310 python infer.py --profile v6p5_cuda_pref --device cuda --models v6p5_cuda_pref_smoke300 --runs 3 --out v6p5_cuda_pref_smoke300`
- full：
  - `conda run -n ros2py310 python train.py --profile v6p5_cuda_pref --device cuda --out v6p5_cuda_pref_full3000`
  - `conda run -n ros2py310 python infer.py --profile v6p5_cuda_pref --device cuda --models v6p5_cuda_pref_full3000 --runs 20 --out v6p5_cuda_pref_full3000`

## 代表 run
- 训练：`runs/v6p5_bn_quick20/train_20260219_165821`（quick20）
- 推理：`runs/v6p5_bn_quick20/train_20260219_165821/infer/20260219_170130`（runs=3）

## 当前结论
- 已完成“局部高分辨率观测 + CNN BatchNorm”代码改造与配置落地。
- 已完成 quick20 先验试跑并产出首批 KPI，但该口径不替代标准 smoke/full 结论。

## 下一步
1. 先执行 `self-check -> smoke`，确认训练稳定性与观测行为。
2. 若 smoke 有正向信号，再进入 `runs=20` 的 full。
3. 将 `run_dir/run_json/kpi` 与 `failure_reason` 回填到本版本四件套。

# v8p16（B 线：训练侧）— 观测加入 progress distance-field map（occ + dist）

## 0) 版本目标

- 在 **SR≈1.0** 前提下压 `avg_path_length` / `path_time_s`，并在 **同一 fixed pairs** 下与 baseline（`Hybrid A*-MPC`）**同跑对比**。
- 禁止作弊：不改 `goal_tolerance_m`（终点容差）及 stop 阈值来“刷 SR”。

## 1) 核心改动（方法摘要）

### 1.1 假设

现有 forest 的全局 CNN 观测仅包含 **静态 occ map**，缺少“到目标的全局 cost-to-go”。在 long 套件中，这可能诱发绕远路（detour）。

### 1.2 做法

在 `AMRBicycleEnv`（bicycle/ackermann 环境）观测中新增一个通道：

- `occ`（障碍占据图，原有）
- `progress-dist`（由 `progress_dist_mode` 计算得到的距离场，下采样到 `obs_map_size`，并归一化映射到 `[-1,1]`）

最终观测：

- `obs = [10 scalars] + [occ map] + [progress-dist map]`
- 观测维度：`10 + 2 * (obs_map_size^2)`（启用开关时）

## 2) 关键配置

- 主入口：`configs/v8p16.json`
- 本地 smoke（episodes=150）：`configs/repro_20260224_v8p16_train_smoke.json`
- fixed pairs3（short/long）：
  - `configs/repro_20260224_v8p16_infer_smoke_short.json`
  - `configs/repro_20260224_v8p16_infer_smoke_long.json`

## 3) 命令（本地回落）

> 说明：`ssh ubuntu-zt` 当前不可用，本轮按“本地回落流程”执行（需在结果留档中记录日期与原因）。

自检：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

训练 smoke：

```bash
conda run -n ros2py310 python train.py --profile repro_20260224_v8p16_train_smoke
```

推理 smoke（fixed pairs3 + baseline 同跑）：

```bash
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p16_infer_smoke_long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p16_infer_smoke_short
```

## 4) 结果摘要（待补）

- short（pairs3）：`SR=1.0`，但 `avg_path_length=18.6356`、`path_time_s=11.2667`（baseline：`16.3207`、`9.4667`）
- long（pairs3）：`SR=1.0`，但 `avg_path_length=39.9377`、`path_time_s=25.4167`（baseline：`32.2801`、`17.4333`）
- 结论：NO-GO（短/长均落后 baseline；不进入 full gate）

## 5) 下一步

- 进入 C 线（结构/算法变种，例如 Distributional/Noisy 等 DQN 变种），以“本地 smoke → fixed pairs3 short/long + baseline 同跑”最快闭环。

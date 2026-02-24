# v8p8 版本说明（Dueling + GlobalCNN-Fusion + 可行性辅助监督；目标：C 门槛）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p7`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**已运行 smoke（episodes=150；runs=3）：NO-GO；full gate（C）待跑**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值），避免隐性放宽到达判定。

## 方法摘要

本版在 DQN 家族内做“更强表征 + 更稳可行性”的组合（不引入 SAC/DDPG）：

1) `dueling`（Dueling DQN 的 V/A 分支合成 Q）  
2) `cnn_backbone=globalcnn_fusion` + `cnn_global_spatial_prior=true`（更强全局地图表征 + agent/goal heatmap 先验）  
3) `aux_admissibility_lambda>0`（训练期可行性辅助监督：用 admissible action mask 做 BCE；推理期仍按既有策略口径执行）  
4) 保留 `v8p7` 的 `forest_goal_approach_override`（接近目标速度整形），并通过固定 pairs 的 full gate 决定是否保留/如何调参。

## 本轮关键命令（计划执行）

### 1) 最小自检

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

### 2) smoke（episodes=150, runs=3）

```bash
conda run -n ros2py310 python train.py --profile repro_20260224_v8p8_smoke
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p8_smoke
```

### 3) full gate（C：short/long，各 runs=20，fixed pairs）

```bash
# short
conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p8_full20_pairs_short

# long
conda run -n ros2py310 python infer.py --profile v8p8 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p8_full20_pairs_long
```

## 代表 run

- smoke：
  - train_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059`
  - infer_run：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556`
  - kpi_mean_raw：`runs/v8p8_dueling_globalcnn_aux_smoke/train_20260224_105059/infer/20260224_110556/table2_kpis_mean_raw.csv`
- 推理侧消融（固定 pairs3）：
  - short：`runs/v8p8_ablate_pairs3_short_r15_sf0p9/20260224_111755/table2_kpis_mean_raw.csv`
  - long：`runs/v8p8_ablate_pairs3_long_r15_sf0p9/20260224_111905/table2_kpis_mean_raw.csv`
- full20（pairs20）：`N/A`

## 结论（待回填）

- smoke 结论：**NO-GO**（short `success_rate` 低于 Hybrid A*-MPC；mid/long 的 `avg_path_length/path_time_s` 显著劣于 Hybrid A*-MPC）。
- 推理侧消融：`forest_goal_admissible_relax_factor` 在当前固定 pairs3 上未见明显改善；后续优先考虑训练侧（reward/目标优化）或推理侧更强的“提速/缩路”策略，但必须以 full gate C（pairs20 + runs=20）验证为准。

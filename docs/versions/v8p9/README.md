# v8p9 版本说明（推理侧 sweep 优先；目标：SR≈1.0 前提下压 avg_path_length / path_time_s）

- 版本类型：**Patch（p+1）**
- 上一版本：`v8p8`
- 上一稳定对照：`v7p1`
- 本版口径：`shielded/hybrid`（`forest_no_fallback=false`）
- 状态：**进行中（推理侧 sweep smoke → full gate C）**

## 本版目标（硬门槛：C）

在 `short/long` 双套件、各 `runs=20` 条件下（固定 pairs，避免 sample drift），至少同时满足：
- `success_rate(CNN-DDQN) >= success_rate(Hybrid A*-MPC)`
- `avg_path_length(CNN-DDQN) < avg_path_length(Hybrid A*-MPC)`
- `path_time_s(CNN-DDQN) < path_time_s(Hybrid A*-MPC)`

反作弊约束：
- **不允许改 `goal_tolerance_m`（终点容差）**；本版也默认不改 `goal_stop_speed_m_s/goal_stop_delta_deg`（停止/摆正阈值），避免隐性放宽到达判定。

## 方法摘要（本版主线）

本版先把“推理侧口径”压到尽可能短/快（维持 SR），再决定是否值得进一步训练侧大改：

- 推理侧 sweep 关键阈值：
  - `forest_min_progress_m`（最小进度阈值）
  - `forest_min_od_m`（最小障碍距离阈值）
  - `forest_replace_topq`（替换候选 top-Q 约束）
  - `forest_topk_turn_penalty`（替换候选的转向惩罚）
  - `forest_goal_approach_speed_factor`（近目标速度整形强度）

## 本轮关键命令（计划）

### 1) 最小自检

```bash
conda run -n ros2py310 python infer.py --self-check
```

### 2) 推理侧 sweep smoke（fixed pairs3，runs=3）

> 说明：infer 的 `--rand-pairs-json` 不能同时对 short/long 指定两份文件，因此 short/long 必须分别跑。

```bash
# short
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_short_smoke

# long
conda run -n ros2py310 python infer.py --profile repro_20260224_v8p9_infer_sweep_long_smoke
```

### 3) full gate（C：short/long，各 runs=20，fixed pairs20）

```bash
# short
conda run -n ros2py310 python infer.py --profile v8p9 \
  --envs forest_a::short --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_short20_v1.json \
  --out v8p9_full20_pairs_short

# long
conda run -n ros2py310 python infer.py --profile v8p9 \
  --envs forest_a::long --runs 20 --random-start-goal \
  --rand-pairs-json configs/repro_20260210_forest_a_pairs_long20_v1.json \
  --out v8p9_full20_pairs_long
```

## 代表 run

- smoke：`N/A`
- full20（pairs20）：`N/A`

## 结论（待回填）

- `N/A`


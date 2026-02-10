# <version>

> 请按目录方式创建：`docs/versions/<version>/`。
> 必备文件：`README.md`、`CHANGES.md`、`RESULTS.md`、`runs/README.md`。

## `README.md`

- 版本类型：**Major (v+1) / Minor (p+1)**
- 研究路线：`CNN-DDQN`（推理口径：`strict-argmax` / `shielded/masked/hybrid`；命名必须与实现一致）
- 状态：**未通过 / 通过**
- 上一版本：`<prev_version>`

### 概要
- 一句话说明本版目标。
- 一句话总结 short/long 对比基线结果。
- 一句话给出决策（继续/回退/升级）。

### 方法意图
- 说明本版主要要改善的问题（如 超时崩塌、long 套件推进）。
- 说明该改动为何仍满足算法定义一致性。

### 复现实验配置 / 命令
- 配置文件：`configs/<profile_or_repro>.json`
- 命令：
  - `conda run -n ros2py310 python train.py --profile <profile>`
  - `conda run -n ros2py310 python infer.py --profile <profile>`

### 代表性运行
- run_dir: `runs/<...>/train_<...>`
- run_json: `runs/<...>/configs/run.json`
- KPI 路径： `runs/<...>/infer/<...>/table2_kpis_mean_raw.csv`

### 结论 / 下一步
- 写明当前决策与下一版本目标。

## `CHANGES.md`

### 版本意图
- 简述本版本存在的原因。

### 相对上一版的具体变更
- `<param_or_logic_1>`: `<old>` -> `<new>`
- `<param_or_logic_2>`: `<old>` -> `<new>`

### 变更文件
- `path/to/file1`
- `path/to/file2`

### 关键参数快照
- `k1`: `v1`
- `k2`: `v2`

## `RESULTS.md`

### 代表性正式指标
- KPI 路径： `runs/<...>/table2_kpis_mean_raw.csv`
- CNN short/long 成功率： `<short_sr>` / `<long_sr>`
- Hybrid short/long 成功率： `<base_short_sr>` / `<base_long_sr>`
- CNN 规划代价 short/long：`<short_cost>` / `<long_cost>`
- CNN 路径时间(s) short/long：`<short_time>` / `<long_time>`
- CNN argmax 不可行动作率 short/long：`<short_inad>` / `<long_inad>`

### 门槛检查
- `success_rate(CNN) >= success_rate(Hybrid)`: short `<通过/未通过>`, long `<通过/未通过>`
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: `<通过/未通过/N/A>`
- `path_time_s(CNN) < path_time_s(Hybrid)`: `<通过/未通过/N/A>`
- 最终门槛状态：`<通过/未通过>`

### 失败分析
- short: `<timeout/collision/...>`
- long: `<timeout/collision/...>`
- 主要瓶颈假设。

## `runs/README.md`

### 代表性运行
- run_dir: `runs/<...>`
- run_json: `runs/<...>/configs/run.json`
- kpi: `runs/<...>/infer/<...>/table2_kpis_mean_raw.csv`

### 本版本已知运行
- `runs/<...>`

# v4p1 - 结果

## 代表性正式指标
- KPI 路径：`runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524/table2_kpis_mean_raw.csv`
- CNN short/long 成功率：`0.1` / `0.0`
- Hybrid short/long 成功率：`0.9` / `1.0`
- CNN 规划代价 short/long：`108.375` / `inf`
- CNN 路径时间(s) short/long：`9.15` / `N/A`
- CNN argmax 不可行动作率 short/long：`0.483` / `0.490`

## 门槛检查（对标 Hybrid A*-MPC）
- `success_rate(CNN) >= success_rate(Hybrid)`: short `未通过`（0.1 < 0.9）, long `未通过`（0.0 < 1.0）
- `avg_path_length(CNN) < avg_path_length(Hybrid)`: short `通过`（10.7925 < 15.5051）, long `N/A`（CNN long 无成功轨迹）
- `path_time_s(CNN) < path_time_s(Hybrid)`: short `未通过`（9.15 > 9.0556）, long `N/A`
- 最终门槛状态：`未通过`

## 运行过程记录（同轮次）

### 2026-02-10 已执行自检
- command:
  - `conda run -n ros2py310 python train.py --self-check`
  - `conda run -n ros2py310 python infer.py --self-check`
- 结果：`PASS`（CUDA 可用，`device_ok=cuda:0`）

### 2026-02-10 中止运行（失败归档）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke/train_20260210_133253`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke/train_20260210_133253/configs/run.json`
- kpi: `N/A`
- failure_reason: `manual_interrupt_for_faster_smoke_iteration`

### 2026-02-10 中止运行（失败归档）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 320 --max-steps 600 --train-eval-every 20 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e320_ms600`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e320_ms600/train_20260210_133728`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e320_ms600/train_20260210_133728/configs/run.json`
- kpi: `N/A`
- failure_reason: `manual_interrupt_for_faster_smoke_iteration`

### 2026-02-10 中止运行（失败归档）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 300 --max-steps 600 --forest-demo-pretrain-steps 12000 --train-eval-every 20 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e300_fast`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e300_fast/train_20260210_133947`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e300_fast/train_20260210_133947/configs/run.json`
- kpi: `N/A`（训练未完成）
- failure_reason: `manual_interrupt_for_time_budget`

### 2026-02-10 quick smoke（无 demo 预填充）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 8 --max-steps 600 --learning-starts 100 --no-forest-demo-prefill --forest-demo-pretrain-steps 0 --train-eval-every 0 --eval-every 0 --progress --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo/train_20260210_134834/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo/train_20260210_134834`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo/train_20260210_134834/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_quick_nodemo_infer10/20260210_135006/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`。

### 2026-02-10 quick smoke（iter2，demo4k）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k/train_20260210_135138/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter2_demo4k_infer10/20260210_135524/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.1/0.0`（short 有轻微恢复，long 仍 0）。

### 2026-02-10 中止运行（失败归档）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 120 --max-steps 1200 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e120_ms1200_fix`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e120_ms1200_fix/train_20260210_140143`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_e120_ms1200_fix/train_20260210_140143/configs/run.json`
- kpi: `N/A`
- failure_reason: `manual_interrupt_for_time_budget`

## 失败分析（基于 v4p1 quick smoke 原始 KPI）
- short: `collision=9`, `reached=1`
- long: `collision=10`
- 主要瓶颈：`argmax_inadmissible_rate` 偏高（short/long ≈ `0.483/0.490`），碰撞主导且 long 仍未形成可达策略。
- 结论：本版机制已落地并可运行，但性能门槛远未满足，需继续 v4p2 调参与结构迭代。

## 运行过程增量（2026-02-10，iter3~iter5）

### 2026-02-10 quick smoke（iter3，dynamic-fix 验证）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix/train_20260210_140910/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix/train_20260210_140910`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix/train_20260210_140910/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter3_dynamicfix_infer10/20260210_141327/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.1/0.0`；`argmax_inad(short/long)=0.468/0.491`。
- failure_reason（raw）：short `collision=9,reached=1`；long `collision=10`。

### 2026-02-10 quick smoke（iter4，单变量：pretrain steps=12000）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 12000 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k/train_20260210_141512/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k/train_20260210_141512`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k/train_20260210_141512/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter4_pretrain12k_infer10/20260210_142121/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`（明显退化）；`argmax_inad(short/long)=0.565/0.386`。
- failure_reason（raw）：short `collision=7,timeout=3`；long `collision=7,timeout=3`。

### 2026-02-10 quick smoke（iter5，单变量：delta_dot_bins=9）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --forest-action-delta-dot-bins 9 --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9/train_20260210_142313/models --forest-action-delta-dot-bins 9 --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9/train_20260210_142313`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9/train_20260210_142313/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter5_ddot9_infer10/20260210_142805/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.1/0.0`；`argmax_inad(short/long)=0.459/0.372`。
- failure_reason（raw）：short `collision=6,reached=1,timeout=3`；long `collision=7,timeout=3`。

## 增量结论（iter3~iter5）
- 当前最佳成功率仍是 `short=0.1,long=0.0`（iter2/iter3/iter5 持平）。
- `iter4`（pretrain 12000）出现明显回退（`0.0/0.0`），说明仅增加示教预训练步数不能稳定带来提升。
- `iter5` 虽将 long `argmax_inadmissible_rate` 降到 `0.372`，但 long 成功率仍 `0.0`，主失败从“纯 collision”转为“collision+timeout”。
- v4p1 截止当前仍未通过最终门槛，需继续迭代。

### 2026-02-10 quick smoke（iter6，单变量：关闭动态课程）
- command:
  - `conda run -n ros2py310 python train.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --episodes 40 --max-steps 600 --learning-starts 300 --forest-demo-target-mult 4 --forest-demo-target-cap 4000 --forest-demo-pretrain-steps 4000 --no-forest-train-dynamic-curriculum --train-eval-every 10 --eval-every 0 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic`
  - `conda run -n ros2py310 python infer.py --profile repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke --models runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic/train_20260210_143158/models --runs 10 --out repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic_infer10`
- run_dir: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic/train_20260210_143158`
- run_json: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic/train_20260210_143158/configs/run.json`
- kpi: `runs/repro_20260210_forest_a_cnn_ddqn_strict_no_fallback_v4p1_smoke_iter6_nodynamic_infer10/20260210_143711/table2_kpis_mean_raw.csv`
- 关键结果：CNN `short/long=0.0/0.0`；`argmax_inad(short/long)=0.531/0.407`。
- failure_reason（raw）：short `collision=10`；long `collision=10`。

## 增量结论更新（含 iter6）
- `iter6` 关闭动态课程后并未恢复 `v3p11` 表现，反而回到 `0.0/0.0`。
- 截至 2026-02-10 本轮迭代，`v4p1` 最优仍为 `short=0.1,long=0.0`，未出现 long 非零样本。

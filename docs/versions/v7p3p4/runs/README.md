# v7p3p4 runs 追溯

## 1) 本轮执行命令（实际）
- 本地 -> 远端同步：
  - `rsync -avz --delete --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/`
- 远端 self-check：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"`
- 远端 infer smoke（固定模型，不重训）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p4_safe_fallback_infer_smoke"`
- 远端 long-only 推理侧消融（固定模型，不重训；目标：压 `avg_path_length`）：
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p4_safe_fallback_infer_smoke --envs forest_a::long --out v7p3p4_pathlen_sweep_long_S1 --forest-adm-horizon 10 --forest-min-od-m 0.0 --forest-min-progress-m -0.2 --forest-topk-turn-penalty 0.0 --device cuda --progress"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p4_safe_fallback_infer_smoke --envs forest_a::long --out v7p3p4_pathlen_sweep_long_S2 --forest-adm-horizon 15 --forest-min-od-m 0.0 --forest-min-progress-m -0.2 --forest-topk-turn-penalty 0.0 --device cuda --progress"`
  - `ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260222_v7p3p4_safe_fallback_infer_smoke --envs forest_a::long --out v7p3p4_pathlen_sweep_long_S3 --forest-adm-horizon 10 --forest-min-od-m 0.0 --forest-min-progress-m 0.0 --forest-topk-turn-penalty 0.0 --device cuda --progress"`
- 远端 -> 本地回传：
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p4_safe_fallback_infer_smoke/ /home/sun/phdproject/dqn/dqn/runs/v7p3p4_safe_fallback_infer_smoke/`
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S1/ /home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S1/`
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S2/ /home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S2/`
  - `rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S3/ /home/sun/phdproject/dqn/dqn/runs/v7p3p4_pathlen_sweep_long_S3/`

## 2) 固定模型来源（本轮不重训）
- `models_run_dir`：`runs/v7p3p2_turnaware_smoke/train_20260222_101744`

## 3) run 路径登记
- infer：
  - `run_dir`：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513`
  - `run_json`：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/configs/run.json`
  - `kpi_mean_raw`：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_mean_raw.csv`
  - `kpi_raw`：`runs/v7p3p4_safe_fallback_infer_smoke/20260222_141513/table2_kpis_raw.csv`
- infer（long-only sweep）：
  - S1：
    - `run_dir`：`runs/v7p3p4_pathlen_sweep_long_S1/20260222_172355`
    - `run_json`：`runs/v7p3p4_pathlen_sweep_long_S1/20260222_172355/configs/run.json`
    - `kpi_mean_raw`：`runs/v7p3p4_pathlen_sweep_long_S1/20260222_172355/table2_kpis_mean_raw.csv`
    - `kpi_raw`：`runs/v7p3p4_pathlen_sweep_long_S1/20260222_172355/table2_kpis.csv`
  - S2：
    - `run_dir`：`runs/v7p3p4_pathlen_sweep_long_S2/20260222_172426`
    - `run_json`：`runs/v7p3p4_pathlen_sweep_long_S2/20260222_172426/configs/run.json`
    - `kpi_mean_raw`：`runs/v7p3p4_pathlen_sweep_long_S2/20260222_172426/table2_kpis_mean_raw.csv`
    - `kpi_raw`：`runs/v7p3p4_pathlen_sweep_long_S2/20260222_172426/table2_kpis.csv`
  - S3：
    - `run_dir`：`runs/v7p3p4_pathlen_sweep_long_S3/20260222_172453`
    - `run_json`：`runs/v7p3p4_pathlen_sweep_long_S3/20260222_172453/configs/run.json`
    - `kpi_mean_raw`：`runs/v7p3p4_pathlen_sweep_long_S3/20260222_172453/table2_kpis_mean_raw.csv`
    - `kpi_raw`：`runs/v7p3p4_pathlen_sweep_long_S3/20260222_172453/table2_kpis.csv`

## 4) 关键参数快照
- 复现配置：`configs/repro_20260222_v7p3p4_safe_fallback_infer_smoke.json`
- 推理关键参数：
  - `forest_no_fallback=false`（`shielded/hybrid`）
  - `forest_adm_horizon=30`
  - `forest_topk=10`
  - `forest_min_od_m=0.02`
  - `forest_min_progress_m=0.0`
  - `forest_topk_turn_penalty=0.3`

## 5) 关键 debug 指标（来自 `table2_kpis_mean_raw.csv`）
- short：`argmax_inadmissible_rate=0.192`，`fallback_rate=0.192`
- mid：`argmax_inadmissible_rate=0.470`，`fallback_rate=0.470`
- long：`argmax_inadmissible_rate=0.327`，`fallback_rate=0.327`

## 6) `failure_reason` 分布（CNN-DDQN）
- short：`reached=2`，`timeout=1`
- mid：`reached=2`，`timeout=1`
- long：`reached=3`
- 合计：`reached=7`，`timeout=2`（`collision=0`）

## 7) long-only sweep `failure_reason` 分布（CNN-DDQN）
- S1：`collision=3`
- S2：`timeout=2`，`collision=1`
- S3：`collision=2`，`timeout=1`

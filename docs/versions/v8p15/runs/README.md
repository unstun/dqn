# v8p15 runs 索引与执行记录

- 执行位置：`ubuntu-zt`（远端优先）
- 远端环境：`conda run -n ros2py310`
- 反作弊：不改 `goal_tolerance_m`；baseline 必须与 RL 同跑；对比必须使用 fixed pairs（避免 sample drift）。

## 0) 远端同步口径（本地覆盖远端，不包含 runs/）

```bash
rsync -avz --delete \
  --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' \
  /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

## 1) infer sweep（long，fixed pairs3，runs=3，baseline 同跑）

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p2"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p3"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p15_infer_sweep_long_pairs3_sigma0p5"
```

- run_dir（sigma=0.2）：`runs/v8p15_sweep_long_pairs3_sigma0p2/20260224_180451`
- run_dir（sigma=0.3）：`runs/v8p15_sweep_long_pairs3_sigma0p3/20260224_180520`
- run_dir（sigma=0.5）：`runs/v8p15_sweep_long_pairs3_sigma0p5/20260224_180550`

## 2) 结果回传（仅回传对应 out 目录）

```bash
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p2/ /home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p2/
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p3/ /home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p3/
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p5/ /home/sun/phdproject/dqn/dqn/runs/v8p15_sweep_long_pairs3_sigma0p5/
```

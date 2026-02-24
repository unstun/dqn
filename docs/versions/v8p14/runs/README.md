# v8p14 runs 索引与执行记录

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
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg002"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p14_infer_sweep_long_pairs3_w1p5_mpneg005"
```

- run_dir（mp=-0.02）：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg002/20260224_174532`
- run_dir（mp=-0.05）：`runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/20260224_174542`

## 2) 结果回传（仅回传对应 out 目录）

```bash
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p14_sweep_long_pairs3_w1p5_mpneg002/ /home/sun/phdproject/dqn/dqn/runs/v8p14_sweep_long_pairs3_w1p5_mpneg002/
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/ /home/sun/phdproject/dqn/dqn/runs/v8p14_sweep_long_pairs3_w1p5_mpneg005/
```

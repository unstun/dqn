# v8p13 runs 索引与执行记录

- 执行位置：`ubuntu-zt`（远端优先）
- 远端环境：`conda run -n ros2py310`
- 反作弊：不改 `goal_tolerance_m`；baseline 必须与 RL 同跑；对比必须使用 fixed pairs（避免 sample drift）。

## 0) 远端同步口径（本地覆盖远端，不包含 runs/）

```bash
rsync -avz --delete \
  --exclude '.git/' --exclude 'runs/' --exclude '__pycache__/' --exclude '*.pyc' \
  /home/sun/phdproject/dqn/dqn/ ubuntu-zt:/home/sun/phdproject/dqn/dqn/
```

## 1) 远端最小自检

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --self-check"
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --self-check"
```

## 2) train smoke（episodes=150）

```bash
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python train.py --profile repro_20260224_v8p13_train_smoke"
```

- run_dir：`runs/v8p13/train_20260224_170708`

## 3) infer smoke（fixed pairs3，runs=3，baseline 同跑）

```bash
# short
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p13_infer_smoke_short"

# long
ssh ubuntu-zt "cd /home/sun/phdproject/dqn/dqn && /home/sun/miniconda3/bin/conda run -n ros2py310 python infer.py --profile repro_20260224_v8p13_infer_smoke_long"
```

- short run_dir：`runs/v8p13_infer_smoke_short_pairs3/20260224_172126`
- long run_dir：`runs/v8p13_infer_smoke_long_pairs3/20260224_172037`

## 4) 结果回传（仅回传对应 out 目录）

```bash
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p13/ /home/sun/phdproject/dqn/dqn/runs/v8p13/
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p13_infer_smoke_short_pairs3/ /home/sun/phdproject/dqn/dqn/runs/v8p13_infer_smoke_short_pairs3/
rsync -avz ubuntu-zt:/home/sun/phdproject/dqn/dqn/runs/v8p13_infer_smoke_long_pairs3/ /home/sun/phdproject/dqn/dqn/runs/v8p13_infer_smoke_long_pairs3/
```

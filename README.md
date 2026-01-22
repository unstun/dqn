# AMR DQN (I)DDQN Repro

## Repository layout

- `amr_dqn/`: core library code
  - `amr_dqn/maps/`: map specifications
  - `amr_dqn/cli/`: training + inference CLIs
- `train.py`, `infer.py`: thin wrappers (keep old commands working)
- `runs/`: all generated results (experiments + timestamped runs)
- `docs/`: paper + extracted figures/assets

## Conda (CUDA)

GPU training/inference requires a CUDA-enabled PyTorch build. In this repo, the recommended environment is:

```powershell
D:/anaconda/Scripts/activate
conda activate ros2py310
python train.py --self-check --device cuda
```

Non-interactive alternative (no manual activation):

```powershell
conda run -n ros2py310 python train.py --self-check --device cuda
```

## Training

```
python train.py --out outputs_repro_1000 --episodes 1000 --device cuda
```

Outputs go to `runs/outputs_repro_1000/train_<timestamp>/...` (models, curves, configs).

## Forest scenario (bicycle model, 0.1m)

Train on generated forest maps (Ackermann/bicycle dynamics, 35 discrete `(δ̇, a)` actions):

```
python train.py --envs forest_a forest_b forest_c forest_d --out outputs_forest --episodes 1000 --device cuda
```

## Inference / KPIs

Use the latest training run for an experiment name:

```
python infer.py --models outputs_repro_1000 --out outputs_repro_1000 --device cuda
```

Include classical baselines (Hybrid A* + RRT*):

```
python infer.py --models outputs_repro_1000 --out outputs_repro_1000 --baselines all --device cuda
```

Baseline-only (no checkpoints required):

```
python infer.py --envs forest_b --out outputs_forest_baselines --baselines all --skip-rl --max-steps 600 --device cpu
```

Inference outputs are stored under the selected training run:

`runs/outputs_repro_1000/train_<timestamp>/infer/<timestamp>/...`

Or write inference outputs to a separate experiment directory:

```
python infer.py --models outputs_repro_1000 --out outputs_repro_1000_kpi2 --device cuda --cuda-device 0
```

Or point directly at a specific training run directory/models directory:

```
python infer.py --models runs/outputs_repro_1000/train_<timestamp> --out outputs_repro_1000_kpi2
```

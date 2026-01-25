# dqn (forest bicycle env)

本仓库默认只保留森林运动学（Ackermann/bicycle）模型场景：`forest_a` / `forest_b` / `forest_c` / `forest_d`。

## 环境

- 统一使用 conda 环境：`ros2py310`
- 默认使用 CUDA（如需强制 CPU：加 `--device cpu`）

自检（确认 PyTorch/CUDA 可用）：

```bash
conda run -n ros2py310 python train.py --self-check
conda run -n ros2py310 python infer.py --self-check
```

## 训练 / 推理

推荐直接用 profile（见 `configs/*.json`）：

```bash
conda run -n ros2py310 python train.py --profile forest_a_all6_300_cuda
conda run -n ros2py310 python infer.py --profile forest_a_all6_300_cuda
```

更完整的命令示例与参数说明见：`runtxt.md`。

